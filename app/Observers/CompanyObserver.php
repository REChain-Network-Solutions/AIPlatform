<?php

namespace App\Observers;

use App\Enums\MaritalStatus;
use App\Events\NewCompanyCreatedEvent;
use App\Http\Controllers\AppSettingController;
use App\Http\Controllers\CurrencySettingController;
use App\Http\Controllers\RolePermissionController;
use App\Models\AttendanceSetting;
use App\Models\Company;
use App\Models\Currency;
use App\Models\CustomFieldGroup;
use App\Models\DashboardWidget;
use App\Models\DiscussionCategory;
use App\Models\EmailNotificationSetting;
use App\Models\EmployeeShift;
use App\Models\GlobalSetting;
use App\Models\GoogleCalendarModule;
use App\Models\InvoiceSetting;
use App\Models\LeadCustomForm;
use App\Models\LeadSource;
use App\Models\LeadStatus;
use App\Models\LeaveType;
use App\Models\LogTimeFor;
use App\Models\MessageSetting;
use App\Models\Module;
use App\Models\ModuleSetting;
use App\Models\Notification;
use App\Models\PackageUpdateNotify;
use App\Models\Permission;
use App\Models\PermissionRole;
use App\Models\ProjectSetting;
use App\Models\ProjectStatusSetting;
use App\Models\QuickBooksSetting;
use App\Models\Role;
use App\Models\SlackSetting;
use App\Models\SuperAdmin\GlobalCurrency;
use App\Models\SuperAdmin\GlobalInvoice;
use App\Models\SuperAdmin\GlobalSubscription;
use App\Models\SuperAdmin\Package;
use App\Models\SuperAdmin\PackageSetting;
use App\Models\TaskboardColumn;
use App\Models\ThemeSetting;
use App\Models\TicketEmailSetting;
use App\Models\TicketType;
use App\Models\TicketSettingForAgents;
use App\Models\User;
use App\Models\LeadPipeline;
use App\Models\PipelineStage;
use Modules\Aitools\Entities\AiToolsUsageHistory;
use App\Models\TicketChannel;
use App\Models\TicketCustomForm;
use App\Models\UnitType;
use App\Traits\StoreHeaders;
use App\Helper\Files;
use App\Models\UserAuth;
use App\Scopes\ActiveScope;
use App\Scopes\CompanyScope;

class CompanyObserver
{

    use StoreHeaders;

    public function creating(Company $company)
    {
        $this->copyFromGlobalSettings($company);
        $this->dateFormats($company);
        $this->storeHeaders($company);

        $company->leaves_start_from = 'year_start';


        // WORKSUITESAAS
        $this->packageInsert($company);

        if (global_setting()->company_need_approval && !user()?->is_superadmin) {
            $company->approved = 0;
        }
    }

    private function copyFromGlobalSettings($company)
    {
        $globalSetting = global_setting();
        $company->hash = md5(microtime());
        $company->logo_background_color = $globalSetting->logo_background_color == null ? '#ffffff' : $globalSetting->logo_background_color;
        $company->header_color = $globalSetting->header_color;
        $company->login_background = $globalSetting->login_background;
        $company->sidebar_logo_style = $globalSetting->sidebar_logo_style;
        $company->auth_theme = $globalSetting->auth_theme;
        $company->auth_theme_text = $globalSetting->auth_theme_text;
        //        $company->light_logo = $globalSetting->light_logo;
        $company->favicon = $globalSetting->favicon;
        $company->datatable_row_limit = $globalSetting->datatable_row_limit;

        // When company is added from superadmin panel
        if (!user()) {
            $company->timezone = $globalSetting->timezone;
            $company->locale = $globalSetting->locale;
            $company->logo = $globalSetting->logo;
            $company->date_format = $globalSetting->date_format;
            $company->time_format = $globalSetting->time_format;
        }

        return $company;
    }

    public function saving(Company $company)
    {

        $user = user();

        if ($user) {
            $company->last_updated_by = $user->id;
        }


        if ($company->isDirty('approved')) {
            $company->approved_by = $user->id;
        }

        if ($company->isDirty('date_format')) {
            $this->dateFormats($company);
        }

        if ($company->isDirty('currency_id')) {
            (new CurrencySettingController())->updateExchangeRates();
        }

        if (!isRunningInConsoleOrSeeding() && $company->isDirty('currency_id') && !is_null(user())) {
            $allClients = User::allClients();
            $clientsArray = $allClients->pluck('id')->toArray();

            $appSettings = new AppSettingController();
            $appSettings->deleteSessions($clientsArray);
        }

        // IsRunningInConsoleOrSeeding is added to prevent running seeder
        // for the case of running company migration before having global_settings table
        if ($company->id === 1 && isWorksuite() && !isRunningInConsoleOrSeeding()) {
            $global = GlobalSetting::first();
            $global->email = $company->company_email;
            $global->global_app_name = $company->app_name;
            $global->logo_background_color = $company->logo_background_color;
            $global->header_color = $company->header_color;
            $global->login_background = $company->login_background;
            $global->sidebar_logo_style = $company->sidebar_logo_style;
            $global->auth_theme = $company->auth_theme;
            $global->auth_theme_text = $company->auth_theme_text;
            $global->light_logo = $company->light_logo;
            $global->favicon = $company->favicon;
            $global->logo = $company->logo;
            $global->datatable_row_limit = $company->datatable_row_limit;
            $global->timezone = $company->timezone;
            $global->saveQuietly();
        }

        // WORKSUITESAAS
        $this->saasSaving($company);
        cache()->forget('user_' . $company->id . '_is_active');

        session()->forget(['company', 'company.*', 'company.currency', 'company.paymentGatewayCredentials']);
        cache()->forget('global_setting');

    }

    //phpcs:ignore
    public function created(Company $company)
    {
        $this->currencies($company);
        $this->companyAddress($company);
        $this->roles($company);
        $this->employeeShift($company);
        $this->attendanceSetting($company);
        $this->customFieldGroup($company);
        $this->dashboardWidgets($company);
        $this->discussionCategory($company);
        $this->emailNotificationSettings($company);
        $this->invoiceSetting($company);
        $this->leadCustomForms($company);
        $this->leadSources($company);
        $this->leaveType($company);
        $this->logTimeFor($company);
        $this->messageSetting($company);
        $this->projectSetting($company);
        $this->slackSetting($company);
        $this->ticketChannel($company);
        $this->ticketType($company);
        $this->ticketSettingForAgents($company);
        $this->customForms($company);
        $this->taskBoard($company);
        $this->projectStatusSettings($company);

        $company->paymentGatewayCredentials()->create();
        $company->taskSetting()->create();
        $company->leaveSetting()->create();
        $this->dateFormats($company);
        $this->moduleSettings($company);
        $this->themeSetting($company);
        $this->ticketEmailSetting($company);
        $this->googleCalendar($company);
        $this->unitType($company);
        $this->leadStages($company);

        // WORKSUITESAAS
        $this->updateSubscription($company, $company->package);

        // Will be used in various module
        event(new NewCompanyCreatedEvent($company));


    }

    public function deleting(Company $company)
    {
        if ($company->fileStorage->isNotEmpty()) {
            foreach ($company->fileStorage as $files) {
                Files::deleteFile($files->filename, $files->path);
            }
        }

        $this->deleteNotification($company);
    }

    public function deleteNotification($company){
        Notification::whereIn('type', ['App\Notifications\SuperAdmin\NewCompanyRegister', 'App\Notifications\NewUser'])
            ->whereNull('read_at')
            ->where(function ($q) use ($company) {
                $q->where('data', 'like', '{"id":' . $company->id . '%');
                $q->orWhere('data', 'like', '%"company_id":' . $company->id . '%');
            })->delete();
    }


    public function deleted()
    {
        UserAuth::doesntHave('users')->delete();
    }

    public function currencies($company)
    {
        if (isWorksuiteSaas()) {
            $this->globalCurrencyCopy($company);

            return true;
        }

        $currency = new Currency();
        $currency->currency_name = 'Dollars';
        $currency->currency_symbol = '$';
        $currency->currency_code = 'USD';
        $currency->exchange_rate = 1;
        $currency->currency_position = 'left';
        $currency->no_of_decimal = 2;
        $currency->thousand_separator = ',';
        $currency->decimal_separator = '.';
        $currency->company()->associate($company);
        $currency->saveQuietly();

        // Save First currency to company default currency
        $company->currency_id = $currency->id;
        $company->saveQuietly();


        $currency = new Currency();
        $currency->currency_name = 'Pounds';
        $currency->currency_symbol = '£';
        $currency->currency_code = 'GBP';
        $currency->exchange_rate = 1;
        $currency->currency_position = 'left';
        $currency->no_of_decimal = 2;
        $currency->thousand_separator = ',';
        $currency->decimal_separator = '.';
        $currency->company()->associate($company);
        $currency->saveQuietly();

        $currency = new Currency();
        $currency->currency_name = 'Euros';
        $currency->currency_symbol = '€';
        $currency->currency_code = 'EUR';
        $currency->exchange_rate = 1;
        $currency->currency_position = 'left';
        $currency->no_of_decimal = 2;
        $currency->thousand_separator = ',';
        $currency->decimal_separator = '.';
        $currency->company()->associate($company);
        $currency->saveQuietly();

        $currency = new Currency();
        $currency->currency_name = 'Rupee';
        $currency->currency_symbol = '₹';
        $currency->currency_code = 'INR';
        $currency->exchange_rate = 1;
        $currency->currency_position = 'left';
        $currency->no_of_decimal = 2;
        $currency->thousand_separator = ',';
        $currency->decimal_separator = '.';
        $currency->company()->associate($company);
        $currency->saveQuietly();
    }

    public function employeeShift($company)
    {

        $employeeShift = new EmployeeShift();
        $employeeShift->shift_name = 'Day Off';
        $employeeShift->company_id = $company->id;
        $employeeShift->shift_short_code = 'DO';
        $employeeShift->color = '#E8EEF3';
        $employeeShift->office_start_time = '00:00:00';
        $employeeShift->halfday_mark_time = '00:00:00';
        $employeeShift->office_end_time = '00:00:00';
        $employeeShift->late_mark_duration = 0;
        $employeeShift->clockin_in_day = 0;
        $employeeShift->office_open_days = '';
        $employeeShift->saveQuietly();

        $employeeShift = new EmployeeShift();
        $employeeShift->shift_name = 'General Shift';
        $employeeShift->company_id = $company->id;
        $employeeShift->shift_short_code = 'GS';
        $employeeShift->color = '#99C7F1';
        $employeeShift->office_start_time = '09:00:00';
        $employeeShift->halfday_mark_time = '13:00:00';
        $employeeShift->office_end_time = '18:00:00';
        $employeeShift->late_mark_duration = 20;
        $employeeShift->clockin_in_day = 2;
        $employeeShift->office_open_days = '["1","2","3","4","5"]';
        $employeeShift->saveQuietly();
    }

    public function attendanceSetting($company)
    {
        $setting = new AttendanceSetting();
        $setting->company_id = $company->id;
        $setting->office_start_time = '09:00:00';
        $setting->halfday_mark_time = '13:00:00';
        $setting->office_end_time = '18:00:00';
        $setting->late_mark_duration = 20;
        $setting->default_employee_shift = EmployeeShift::where('company_id', $company->id)->where('shift_name', '<>', 'Day Off')->first()->id;
        $setting->alert_after_status = 0;
        $setting->saveQuietly();
    }

    public function customFieldGroup($company)
    {

        $fields = CustomFieldGroup::ALL_FIELDS;

        array_walk($fields, function (&$a) use ($company) {
            $a['company_id'] = $company->id;
        });

        CustomFieldGroup::insert($fields);

    }

    public function dashboardWidgets($company)
    {

        $widgets = DashboardWidget::WIDGETS;

        array_walk($widgets, function (&$a) use ($company) {
            $a['company_id'] = $company->id;
        });

        DashboardWidget::insert($widgets);

    }

    public function discussionCategory($company)
    {
        DiscussionCategory::create([
            'name' => 'General',
            'color' => '#3498DB',
            'company_id' => $company->id
        ]);
    }

    public function emailNotificationSettings($company)
    {

        $notifications = EmailNotificationSetting::NOTIFICATIONS;

        array_walk($notifications, function (&$a) use ($company) {
            $a['company_id'] = $company->id;
        });

        EmailNotificationSetting::insert($notifications);
    }

    public function invoiceSetting($company)
    {

        InvoiceSetting::create([
            'company_id' => $company->id,
            'credit_note_digit' => 3,
            'credit_note_prefix' => 'CN',
            'credit_note_number_separator' => '#',
            'due_after' => 15,
            'estimate_digit' => 3,
            'estimate_prefix' => 'EST',
            'estimate_number_separator' => '#',
            'estimate_terms' => null,
            'gst_number' => null,
            'hsn_sac_code_show' => 0,
            'invoice_digit' => 3,
            'invoice_prefix' => 'INV',
            'invoice_number_separator' => '#',
            'invoice_terms' => 'Thank you for your business.',
            'locale' => 'en',
            'logo' => null,
            'reminder' => null,
            'send_reminder' => 0,
            'send_reminder_after' => 0,
            'show_client_company_address' => 'yes',
            'show_client_company_name' => 'yes',
            'show_client_email' => 'yes',
            'show_client_name' => 'yes',
            'show_client_phone' => 'yes',
            'show_gst' => 'no',
            'show_project' => 0,
            'tax_calculation_msg' => 0,
            'template' => 'invoice-5',
        ]);

        QuickBooksSetting::create(['status' => 0, 'company_id' => $company->id]);
    }

    public function leadCustomForms($company)
    {
        $data = LeadCustomForm::FORM_FIELDS;
        array_walk($data, function (&$a) use ($company) {
            $a['company_id'] = $company->id;
        });
        LeadCustomForm::insert($data);

    }

    public function leadSources($company)
    {
        $sources = [
            ['type' => __('app.email'), 'company_id' => $company->id],
            ['type' => __('app.google'), 'company_id' => $company->id],
            ['type' => __('app.facebook'), 'company_id' => $company->id],
            ['type' => __('app.friend'), 'company_id' => $company->id],
            ['type' => __('app.direct'), 'company_id' => $company->id],
            ['type' => __('app.tv'), 'company_id' => $company->id]
        ];

        LeadSource::insert($sources);

        $status = [
            ['type' => 'pending', 'priority' => 1, 'default' => 1, 'label_color' => '#FFE700', 'company_id' => $company->id],
            ['type' => 'in process', 'priority' => 2, 'default' => 0, 'label_color' => '#009EFF', 'company_id' => $company->id],
            ['type' => 'done', 'priority' => 3, 'default' => 0, 'label_color' => '#1FAE07', 'company_id' => $company->id]
        ];

        LeadStatus::insert($status);
    }

    public function leadStages($company)
    {
        $pipeline = new LeadPipeline();
        $pipeline->name = 'Sales Pipeline';
        $pipeline->company_id = $company->id;
        $pipeline->label_color = '#009EFF';
        $pipeline->default = 1;
        $pipeline->priority = 1;
        $pipeline->save();

        $pipelineStages = [
            ['name' => 'Generated', 'slug' => 'generated', 'lead_pipeline_id' => $pipeline->id, 'priority' => 1, 'default' => 1, 'label_color' => '#FFD700', 'company_id' => $company->id],
            ['name' => 'Qualified', 'slug' => 'qualified', 'lead_pipeline_id' => $pipeline->id, 'priority' => 2, 'default' => 0, 'label_color' => '#009EFF', 'company_id' => $company->id],
            ['name' => 'Initial Contact', 'slug' => 'initial-contact', 'lead_pipeline_id' => $pipeline->id, 'priority' => 3, 'default' => 0, 'label_color' => '#00CED1', 'company_id' => $company->id],
            ['name' => 'Schedule Appointment', 'slug' => 'schedule-appointment', 'lead_pipeline_id' => $pipeline->id, 'priority' => 4, 'default' => 0, 'label_color' => '#32CD32', 'company_id' => $company->id],
            ['name' => 'Proposal Sent', 'slug' => 'proposal-sent', 'lead_pipeline_id' => $pipeline->id, 'priority' => 5, 'default' => 0, 'label_color' => '#FFA07A', 'company_id' => $company->id],
            ['name' => 'Win', 'slug' => 'win', 'lead_pipeline_id' => $pipeline->id, 'priority' => 6, 'default' => 0, 'label_color' => '#1FAE07', 'company_id' => $company->id],
            ['name' => 'Lost', 'slug' => 'lost', 'lead_pipeline_id' => $pipeline->id, 'priority' => 7, 'default' => 0, 'label_color' => '#DB1313', 'company_id' => $company->id]
        ];

        PipelineStage::insert($pipelineStages);

        return $pipeline;

    }

    public function leaveType($company)
    {
        $gender = ['male', 'female', 'others'];
        $maritalStatus = MaritalStatus::toArray();
        $roles = Role::where('name', '<>', 'client')->where('company_id', $company->id)->pluck('id')->toArray();
        $currentTimestamp = now();

        $status = [
            [
                'type_name' => 'Casual',
                'color' => '#16813D',
                'company_id' => $company->id,
                'gender' => json_encode($gender),
                'marital_status' => json_encode($maritalStatus),
                'role' => json_encode($roles),
                'unused_leave' => 'carry forward',
                'leavetype' => 'yearly',
                 'created_at' => $currentTimestamp,
            ],
            [
                'type_name' => 'Sick',
                'color' => '#DB1313',
                'company_id' => $company->id,
                'gender' => json_encode($gender),
                'marital_status' => json_encode($maritalStatus),
                'role' => json_encode($roles),
                'unused_leave' => 'carry forward',
                'leavetype' => 'yearly',
                'created_at' => $currentTimestamp,
            ],
            [
                'type_name' => 'Earned',
                'color' => '#B078C6',
                'company_id' => $company->id,
                'gender' => json_encode($gender),
                'marital_status' => json_encode($maritalStatus),
                'role' => json_encode($roles),
                'unused_leave' => 'carry forward',
                'leavetype' => 'yearly',
                'created_at' => $currentTimestamp,
            ],
        ];

        LeaveType::insert($status);

    }

    public function logTimeFor($company)
    {
        $logTimeFor = new LogTimeFor();
        $logTimeFor->company_id = $company->id;
        $logTimeFor->log_time_for = 'project';
        $logTimeFor->saveQuietly();
    }

    public function messageSetting($company)
    {
        $setting = new MessageSetting();
        $setting->company_id = $company->id;
        $setting->allow_client_admin = 'no';
        $setting->allow_client_employee = 'no';
        $setting->saveQuietly();
    }

    public function projectSetting($company)
    {
        $project_setting = new ProjectSetting();
        $project_setting->company_id = $company->id;
        $project_setting->send_reminder = 'no';
        $project_setting->remind_time = 5;
        $project_setting->remind_type = 'days';
        $project_setting->saveQuietly();
    }

    public function slackSetting($company)
    {
        $slack = new SlackSetting();
        $slack->company_id = $company->id;
        $slack->slack_webhook = null;
        $slack->slack_logo = null;
        $slack->saveQuietly();
    }

    public function ticketChannel($company)
    {
        $channels = [
            ['channel_name' => 'Email', 'company_id' => $company->id],
            ['channel_name' => 'Phone', 'company_id' => $company->id],
            ['channel_name' => 'Twitter', 'company_id' => $company->id],
            ['channel_name' => 'Facebook', 'company_id' => $company->id]
        ];

        TicketChannel::insert($channels);
    }

    public function ticketType($company)
    {
        $types = [
            ['type' => 'Bug', 'company_id' => $company->id],
            ['type' => 'Suggestion', 'company_id' => $company->id],
            ['type' => 'Question', 'company_id' => $company->id],
            ['type' => 'Sales', 'company_id' => $company->id],
            ['type' => 'Code', 'company_id' => $company->id],
            ['type' => 'Management', 'company_id' => $company->id],
            ['type' => 'Problem', 'company_id' => $company->id],
            ['type' => 'Incident', 'company_id' => $company->id],
            ['type' => 'Feature Request', 'company_id' => $company->id],
        ];

        TicketType::insert($types);
    }

    public function ticketSettingForAgents($company)
    {
        $types = [
            ['ticket_scope' => 'assigned_tickets', 'company_id' => $company->id],
        ];

        TicketSettingForAgents::insert($types);
    }

    public function customForms($company)
    {
        $fields = ['Name', 'Email', 'Ticket Subject', 'Ticket Description', 'Type', 'Priority', 'Assign Group'];
        $fieldsName = ['name', 'email', 'ticket_subject', 'ticket_description', 'type', 'priority', 'assign_group'];
        $fieldsType = ['text', 'text', 'text', 'textarea', 'select', 'select', 'select'];

        foreach ($fields as $key => $value) {

            TicketCustomForm::create([
                'field_display_name' => $value,
                'field_name' => $fieldsName[$key],
                'field_order' => $key + 1,
                'field_type' => $fieldsType[$key],
                'company_id' => $company->id,
            ]);

        }
    }

    public function companyAddress($company)
    {
        try {
            $company->companyAddress()->create([
                'address' => $company->address ?? $company->company_name,
                'location' => $company->company_name ?? 'Jaipur, India',
                'is_default' => 1,
                'company_id' => $company->id,
            ]);
            // Incase any exception comes
        } catch (\Exception $exception) {
            $company->companyAddress()->create([
                'address' => 'First Address',
                'location' => $company->company_name ?? 'Jaipur, India',
                'is_default' => 1,
                'company_id' => $company->id,
            ]);
        }

    }

    public function roles($company): void
    {
        $adminRole = new Role();
        $adminRole->name = 'admin';
        $adminRole->company_id = $company->id;
        $adminRole->display_name = 'App Administrator'; // optional
        $adminRole->description = 'Admin is allowed to manage everything of the app.'; // optional
        $adminRole->saveQuietly();

        $employeeRole = new Role();
        $employeeRole->name = 'employee';
        $employeeRole->company_id = $company->id;
        $employeeRole->display_name = 'Employee'; // optional
        $employeeRole->description = 'Employee can see tasks and projects assigned to him.'; // optional
        $employeeRole->saveQuietly();

        $clientRole = new Role();
        $clientRole->name = 'client';
        $clientRole->company_id = $company->id;
        $clientRole->display_name = 'Client'; // optional
        $clientRole->description = 'Client can see own tasks and projects.'; // optional
        $clientRole->saveQuietly();

        $allPermissions = Permission::whereHas('module', function ($query) {
            $query->withoutGlobalScopes()->where('is_superadmin', '0');
        })->get();

        // DELETE ALL PERMISSION ROLE OF ABOVE ROLES IF ANY
        PermissionRole::whereIn('role_id', [$adminRole->id, $employeeRole->id, $clientRole->id])->delete();

        $rolePermissionController = new RolePermissionController();
        $rolePermissionController->permissionRole($allPermissions, 'employee', $company->id);
        $rolePermissionController->rolePermissionInsert($allPermissions, $adminRole->id, 'all');
        $rolePermissionController->permissionRole($allPermissions, 'client', $company->id);
    }

    public function taskBoard($company): void
    {
        $columns = [
            ['column_name' => 'Incomplete', 'label_color' => '#d21010', 'priority' => 1, 'slug' => str_slug('Incomplete', '_'), 'company_id' => $company->id],
            ['column_name' => 'To Do', 'label_color' => '#f5c308', 'priority' => 2, 'slug' => str_slug('To Do', '_'), 'company_id' => $company->id],
            ['column_name' => 'Doing', 'label_color' => '#00b5ff', 'priority' => 3, 'slug' => str_slug('Doing', '_'), 'company_id' => $company->id],
            ['column_name' => 'Completed', 'label_color' => '#679c0d', 'priority' => 4, 'slug' => str_slug('Completed', '_'), 'company_id' => $company->id],
            ['column_name' => 'Waiting Approval', 'label_color' => '#000', 'priority' => 5, 'slug' => str_slug('Waiting Approval', '_'), 'company_id' => $company->id],
        ];

        TaskboardColumn::insert($columns);

        $board = TaskboardColumn::where('slug', 'incomplete')
            ->where('company_id', $company->id)
            ->first();

        $company->default_task_status = $board->id;
        $company->saveQuietly();
    }

    public function projectStatusSettings($company): void
    {
        $columns = ProjectStatusSetting::COLUMNS;

        array_walk($columns, function (&$a) use ($company) {
            $a['company_id'] = $company->id;
        });

        ProjectStatusSetting::insert($columns);

    }

    public function dateFormats($company): void
    {
        switch ($company->date_format) {

        case 'm-d-Y':
            $company->date_picker_format = 'mm-dd-yyyy';
            $company->moment_format = 'MM-DD-YYYY';
            break;
        case 'Y-m-d':
            $company->date_picker_format = 'yyyy-mm-dd';
            $company->moment_format = 'YYYY-MM-DD';
            break;
        case 'd.m.Y':
            $company->date_picker_format = 'dd.mm.yyyy';
            $company->moment_format = 'DD.MM.YYYY';
            break;
        case 'm.d.Y':
            $company->date_picker_format = 'mm.dd.yyyy';
            $company->moment_format = 'MM.DD.YYYY';
            break;
        case 'Y.m.d':
            $company->date_picker_format = 'yyyy.mm.dd';
            $company->moment_format = 'YYYY.MM.DD';
            break;
        case 'd/m/Y':
            $company->date_picker_format = 'dd/mm/yyyy';
            $company->moment_format = 'DD/MM/YYYY';
            break;
        case 'Y/m/d':
            $company->date_picker_format = 'yyyy/mm/dd';
            $company->moment_format = 'YYYY/MM/DD';
            break;
        case 'd-M-Y':
            $company->date_picker_format = 'dd-M-yyyy';
            $company->moment_format = 'DD-MMM-YYYY';
            break;
        case 'd/M/Y':
            $company->date_picker_format = 'dd/M/yyyy';
            $company->moment_format = 'DD/MMM/YYYY';
            break;
        case 'd.M.Y':
            $company->date_picker_format = 'dd.M.yyyy';
            $company->moment_format = 'DD.MMM.YYYY';
            break;
        case 'd M Y':
            $company->date_picker_format = 'dd M yyyy';
            $company->moment_format = 'DD MMM YYYY';
            break;
        case 'd F, Y':
            $company->date_picker_format = 'dd MM, yyyy';
            $company->moment_format = 'yyyy-mm-d';
            break;
        case 'd D M Y':
            $company->date_picker_format = 'dd D M yyyy';
            $company->moment_format = 'DD ddd MMM YYYY';
            break;
        case 'D d M Y':
            $company->date_picker_format = 'D dd M yyyy';
            $company->moment_format = 'ddd DD MMMM YYYY';
            break;
        default:
            $company->date_picker_format = 'dd-mm-yyyy';
            $company->moment_format = 'DD-MM-YYYY';
            break;
        }
    }

    public function moduleSettings($company): void
    {
        if (isWorksuiteSaas()) {
            $this->createModuleSettings($company);

            return;
        }

        $data = [
            'admin' => [
                ...ModuleSetting::CLIENT_MODULES, ...ModuleSetting::OTHER_MODULES
            ],
            'employee' => [
                ...ModuleSetting::CLIENT_MODULES, ...ModuleSetting::OTHER_MODULES
            ],
            'client' => [
                ...ModuleSetting::CLIENT_MODULES
            ]
        ];

        $moduleSettings = [];
        $existingModuleSettings = ModuleSetting::where('company_id', $company->id)->get();

        foreach ($data as $type => $moduleList) {
            foreach ($moduleList as $module) {
                $existingModuleSetting = $existingModuleSettings->where('type', $type)->where('module_name', $module)->first();

                if ($existingModuleSetting) {
                    $existingModuleSetting->update([
                        'status' => 'active'
                    ]);

                    continue;
                }

                $moduleSettings[] = [
                    'company_id' => $company->id,
                    'type' => $type,
                    'module_name' => $module,
                    'status' => 'active',
                ];
            }
        }

        ModuleSetting::insert($moduleSettings);
    }

    public function themeSetting($company): void
    {
        $headerColor = '#1d82f5';
        $sidebarColor = '#171F29';
        $sidebarTextColor = '#99A5B5';
        $linkColor = '#F7FAFF';

        // WORKSUITE SAAS
        $globalTheme = ThemeSetting::withoutGlobalScope(CompanyScope::class)->where('panel', 'superadmin')->first();

        if ($globalTheme) {
            $headerColor = $globalTheme->header_color;
            $sidebarColor = $globalTheme->sidebar_color;
            $sidebarTextColor = $globalTheme->sidebar_text_color;
            $linkColor = $globalTheme->link_color;
        }

        $themeSettings = [
            ['panel' => 'admin', 'company_id' => $company->id, 'header_color' => $headerColor, 'sidebar_color' => $sidebarColor, 'sidebar_text_color' => $sidebarTextColor, 'link_color' => $linkColor],
            ['panel' => 'project_admin', 'company_id' => $company->id, 'header_color' => $headerColor, 'sidebar_color' => $sidebarColor, 'sidebar_text_color' => $sidebarTextColor, 'link_color' => $linkColor],
            ['panel' => 'employee', 'company_id' => $company->id, 'header_color' => $headerColor, 'sidebar_color' => $sidebarColor, 'sidebar_text_color' => $sidebarTextColor, 'link_color' => $linkColor],
            ['panel' => 'client', 'company_id' => $company->id, 'header_color' => $headerColor, 'sidebar_color' => $sidebarColor, 'sidebar_text_color' => $sidebarTextColor, 'link_color' => $linkColor],
        ];

        ThemeSetting::insert($themeSettings);
    }

    public function ticketEmailSetting($company): void
    {
        $setting = new TicketEmailSetting();
        $setting->company_id = $company->id;
        $setting->saveQuietly();
    }

    public function googleCalendar($company): void
    {
        $module = new GoogleCalendarModule();
        $module->company_id = $company->id;
        $module->lead_status = 0;
        $module->leave_status = 0;
        $module->invoice_status = 0;
        $module->contract_status = 0;
        $module->task_status = 0;
        $module->event_status = 0;
        $module->holiday_status = 0;
        $module->saveQuietly();
    }

    public function unitType($company): void
    {
        $unitTypes = ['unit_type' => 'Pcs', 'default' => 1, 'company_id' => $company->id];

        UnitType::create($unitTypes);
    }

    // WORKSUITESAAS
    private function packageInsert($company)
    {
        // Package setting for get trial package active or not
        $packageSetting = PackageSetting::where('status', 'active')->first();
        $packages = Package::all();

        // get trial package data
        $trialPackage = $packages->filter(function ($value) {
            return $value->default == 'trial';
        })->first();

        // get default package data
        $defaultPackage = $packages->filter(function ($value) {
            return $value->default == 'yes';
        })->first();

        // get another  package data if trial and default package not found
        $otherPackage = $packages->filter(function ($value) {
            return $value->default == 'no';
        })->first();

        // if trial package is active set package to company
        if ($packageSetting && !is_null($trialPackage)) {
            $company->package_id = $trialPackage->id;
            // set company license expire date
            $noOfDays = (!is_null($packageSetting->no_of_days) && $packageSetting->no_of_days != 0) ? $packageSetting->no_of_days : 30;
            $company->licence_expire_on = now()->addDays($noOfDays)->format('Y-m-d');
        }

        // if trial package is not active set default package to company
        elseif (!is_null($defaultPackage)) {
            $company->package_id = $defaultPackage->id;

        }
        else {
            $company->package_id = $otherPackage->id;
        }
    }

    private function globalCurrencyCopy($company): void
    {
        $currencies = GlobalCurrency::all();

        $data = $currencies->map(function ($currency) use ($company) {
            return [
                'currency_name' => $currency->currency_name,
                'currency_symbol' => $currency->currency_symbol,
                'currency_code' => $currency->currency_code,
                'exchange_rate' => $currency->exchange_rate,
                'currency_position' => $currency->currency_position,
                'no_of_decimal' => $currency->no_of_decimal,
                'thousand_separator' => $currency->thousand_separator,
                'decimal_separator' => $currency->decimal_separator,
                'company_id' => $company->id,
            ];
        })->toArray();

        Currency::insert($data);

        $defaultCurrencyQuery = Currency::where('company_id', $company->id);

        if (global_setting()->currency) {
            $defaultCurrencyQuery->where('currency_code', global_setting()->currency->currency_code);
        }

        $defaultCurrency = $defaultCurrencyQuery->firstOrFail();

        $company->currency_id = $defaultCurrency->id;
        $company->saveQuietly();
    }

    public function clearCompanyUserCache($company): void
    {
        User::withoutGlobalScopes([ActiveScope::class, CompanyScope::class])
            ->where('company_id', $company->id)->each(function ($user) {
                cache()->forget('user_modules_' . $user->id);
            });
    }

    public function createModuleSettings($company): void
    {
        $package = Package::where('id', $company->package_id)->first();
        if (!is_null($package)) {
            $moduleInPackage = collect(json_decode($package->module_in_package));

            $data = [
                'admin' => [
                    ...ModuleSetting::CLIENT_MODULES, ...ModuleSetting::OTHER_MODULES
                ],
                'employee' => [
                    ...ModuleSetting::CLIENT_MODULES, ...ModuleSetting::OTHER_MODULES
                ],
                'client' => [
                    ...ModuleSetting::CLIENT_MODULES
                ]
            ];

            $moduleSettings = [];
            $existingModuleSettings = ModuleSetting::where('company_id', $company->id)->get();

            foreach ($data as $type => $moduleList) {
                foreach ($moduleList as $module) {
                    $existingModuleSetting = $existingModuleSettings->where('type', $type)->where('module_name', $module)->first();

                    if ($existingModuleSetting) {
                        $existingModuleSetting->update([
                            'status' => $moduleInPackage->contains($module) ? 'active' : 'deactive',
                            'is_allowed' => $moduleInPackage->contains($module) ? 1 : 0,
                        ]);

                        continue;
                    }

                    $moduleSettings[] = [
                        'company_id' => $company->id,
                        'type' => $type,
                        'module_name' => $module,
                        'status' => $moduleInPackage->contains($module) ? 'active' : 'deactive',
                        'is_allowed' => $moduleInPackage->contains($module) ? 1 : 0,
                    ];
                }
            }

            ModuleSetting::insert($moduleSettings);

            $this->clearCompanyUserCache($company);
        }

    }

    public function updateModuleSettings($company): void
    {

        $moduleSettings = ModuleSetting::where('company_id', $company->id)->get();
        $moduleInPackage = collect(json_decode(Package::where('id', $company->package_id)->first()->module_in_package));
        self::widgetUpdate($company, $moduleInPackage->toArray());

        $activeModuleSettings = [];
        $inactiveModuleSettings = [];

        foreach ($moduleSettings as $moduleSetting) {
            if ($moduleInPackage->contains($moduleSetting->module_name)) {
                $activeModuleSettings[] = $moduleSetting->id;
            }
            else {
                $inactiveModuleSettings[] = $moduleSetting->id;
            }
        }

        ModuleSetting::whereIn('id', $activeModuleSettings)->update(['is_allowed' => 1, 'status' => 'active']);
        ModuleSetting::whereIn('id', $inactiveModuleSettings)->update(['is_allowed' => 0, 'status' => 'deactive']);

        $this->clearCompanyUserCache($company);
    }

    public function saasSaving(Company $company): void
    {

        if ($company->isDirty('licence_expire_on')) {
            $company->license_updated_at = now();
        }

        if ($company->isDirty('package_id')) {
            $package = Package::where('default', 'no')->where('is_free', 0)->where('id', $company->package_id)->first();

            if ($package) {
                $company->subscription_updated_at = now();
            }

            $this->updateModuleSettings($company);

            PackageUpdateNotify::where('company_id', $company->id)->delete();
            clearCompanyValidPackageCache($company->id);

            // Reset AI Tools usage history when package is updated
            if (module_enabled('Aitools')) {
                AiToolsUsageHistory::where('company_id', $company->id)->delete();
            }
        }

    }

    public static function widgetUpdate($company, $moduleInPackage)
    {
        $modulesAll = Module::where('module_name', '<>', 'settings')
            ->where('module_name', '<>', 'dashboards')
            ->whereNotIn('module_name', Module::disabledModuleArray())
            ->get();
        $moduleArray = $modulesAll->pluck('module_name')->toArray();

        $missingModules = array_diff($moduleArray, $moduleInPackage);
        $commonModules = array_intersect($moduleArray, $moduleInPackage);
        $widgets = DashboardWidget::MODULE;

        foreach ($widgets as $widget) {
            foreach ($widget as $key => $widgetValue) {
                if (in_array($key, $missingModules)) {
                    if (!is_array($widgetValue)) {
                        $widgetValue = [$widgetValue];
                    }

                    DashboardWidget::whereIn('widget_name', $widgetValue)
                        ->where('company_id', $company->id)
                        ->where('dashboard_type', 'private-dashboard')
                        ->update(['active' => 0]);

                }
                elseif (in_array($key, $commonModules)) {
                    if (!is_array($widgetValue)) {
                        $widgetValue = [$widgetValue];

                    }

                    DashboardWidget::whereIn('widget_name', $widgetValue)
                        ->where('company_id', $company->id)
                        ->where('dashboard_type', 'private-dashboard')
                        ->update(['active' => 1]);
                }
            }
        }

    }

    public function updateSubscription(Company $company, Package $package)
    {
        $packageType = $package->annual_status ? 'annual' : 'monthly';
        $currencyId = $package->currency_id ?: global_setting()->currency_id;
        $planExpireDate = $company->licence_expire_on;

        if (!$planExpireDate) {
            $planExpireDate = $packageType == 'annual' ? now()->addYear() : now()->addMonth();
        }

        GlobalSubscription::where('company_id', $company->id)
            ->where('subscription_status', 'active')
            ->update(['subscription_status' => 'inactive']);

        $subscription = new GlobalSubscription();
        $subscription->company_id = $company->id;
        $subscription->package_id = $package->id;
        $subscription->currency_id = $currencyId;
        $subscription->package_type = $packageType;
        $subscription->quantity = 1;
        $subscription->gateway_name = 'offline';
        $subscription->subscription_status = 'active';
        $subscription->subscribed_on_date = now();
        $subscription->ends_at = $planExpireDate;
        $subscription->transaction_id = str(str()->random(15))->upper();
        $subscription->save();

        $offlineInvoice = new GlobalInvoice();
        $offlineInvoice->global_subscription_id = $subscription->id;
        $offlineInvoice->company_id = $company->id;
        $offlineInvoice->currency_id = $currencyId;
        $offlineInvoice->package_id = $company->package_id;
        $offlineInvoice->package_type = $packageType;
        $offlineInvoice->total = 0.00;
        $offlineInvoice->pay_date = now();
        $offlineInvoice->next_pay_date = $planExpireDate;
        $offlineInvoice->gateway_name = 'offline';
        $offlineInvoice->status = 'active';
        $offlineInvoice->transaction_id = $subscription->transaction_id;
        $offlineInvoice->save();
    }

}
