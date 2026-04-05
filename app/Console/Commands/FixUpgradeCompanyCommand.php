<?php

namespace App\Console\Commands;

use App\Events\NewCompanyCreatedEvent;
use App\Models\AttendanceSetting;
use App\Models\Company;
use App\Models\CustomField;
use App\Models\CustomFieldGroup;
use App\Models\DashboardWidget;
use App\Models\EmployeeShift;
use App\Models\GlobalSetting;
use App\Models\LanguageSetting;
use App\Models\PaymentGatewayCredentials;
use App\Models\PusherSetting;
use App\Models\SlackSetting;
use App\Models\SuperAdmin\GlobalPaymentGatewayCredentials;
use App\Observers\CompanyObserver;
use App\Scopes\ActiveScope;
use App\Scopes\CompanyScope;
use Database\Seeders\CoreDatabaseSeeder;
use Database\Seeders\EmployeePermissionSeeder;
use Illuminate\Console\Command;
use Illuminate\Support\Facades\Artisan;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Schema;

class FixUpgradeCompanyCommand extends Command
{

    /**
     * The name and signature of the console command.
     *
     * @var string
     */
    protected $signature = 'fix:company';

    /**
     * The console command description.
     *
     * @var string
     */
    protected $description = 'This will used for all the existing companies data';

    /**
     * Execute the console command.
     *
     * @return int
     * @throws RelatedResourceNotFoundException
     */
    public function handle()
    {
        if (!isWorksuiteSaas()) {
            return true;
        }

        $companyCount = Company::withoutGlobalScope(ActiveScope::class)->count();

        // Do not run below command if company is not present in database
        if ($companyCount == 0) {
            return true;
        }

        // TRUNCATE TABLES
        if (!config('app.non_saas_to_saas_enabled')) {
            DB::table('module_settings')->truncate();
            DB::table('dashboard_widgets')->truncate();
            DB::table('email_notification_settings')->truncate();
            DB::table('database_backup_cron_settings')->truncate();
            DB::table('pusher_settings')->truncate();
            PusherSetting::create();
        }


        Artisan::call('sync-user-permissions all');

        $coreDatabaseSeeder = new CoreDatabaseSeeder();
        $coreDatabaseSeeder->dashboardBackupSetting();

        $this->languageSettings();

        $this->customFieldFix();

        $companies = Company::withoutGlobalScope(ActiveScope::class)->get();

        $companyObserver = new CompanyObserver();

        foreach ($companies as $company) {
            echo $company->id . '' . "\n";

            $this->seedPermission($company->id);
            $companyObserver->moduleSettings($company);
            $this->customFieldGroup($company);
            $this->fixDefaultGatewayStatus($company->id);

            $companyObserver->companyAddress($company);
            $companyObserver->dashboardWidgets($company);
            $companyObserver->discussionCategory($company);
            $companyObserver->emailNotificationSettings($company);
            $companyObserver->projectStatusSettings($company);
            $companyObserver->dateFormats($company);
            $companyObserver->ticketEmailSetting($company);
            $companyObserver->googleCalendar($company);

            $companyObserver->employeeShift($company);

            // Retrieve the first employee shift record with the company id
            $shiftId = EmployeeShift::where('company_id', $company->id)->first()->id;
            // Update the default_employee_shift field in the attendance settings table with the shift id
            AttendanceSetting::where('company_id', $company->id)->update(['default_employee_shift' => $shiftId]);

            // Will be used in various module
            config(['app.seeding' => true]);
            event(new NewCompanyCreatedEvent($company));
            config(['app.seeding' => false]);
        }

        DashboardWidget::query()->update(['status' => 1]);

        // Logo Update All Companies
        $this->logoUpdate();

        // hash add all invoices, projects, etc
        $this->hashGenerate();

        // Fix names of packages
        $this->packageFix();

        $this->fixCustomLeadForm();

        $this->languageFixFront();

        $this->paymentGatewayEnumChange();

        $this->storageSettingFix();

        $this->flushAll();

        return Command::SUCCESS;
    }

    private function fixDefaultGatewayStatus($companyId)
    {
        $paymentGatewayCredentials = PaymentGatewayCredentials::where('company_id', $companyId)->first();
        $paymentGatewayCredentials->paypal_mode = is_null($paymentGatewayCredentials->paypal_client_id) ? 'sandbox' : 'live';
        $paymentGatewayCredentials->stripe_mode = is_null($paymentGatewayCredentials->live_stripe_client_id) ? 'test' : 'live';
        $paymentGatewayCredentials->razorpay_mode = is_null($paymentGatewayCredentials->live_razorpay_key) ? 'test' : 'live';
        $paymentGatewayCredentials->paystack_mode = is_null($paymentGatewayCredentials->paystack_secret) ? 'sandbox' : 'live';
        $paymentGatewayCredentials->payfast_mode = is_null($paymentGatewayCredentials->payfast_merchant_key) ? 'sandbox' : 'live';
        $paymentGatewayCredentials->square_environment = is_null($paymentGatewayCredentials->square_access_token) ? 'sandbox' : 'production';
        $paymentGatewayCredentials->flutterwave_mode = is_null($paymentGatewayCredentials->live_flutterwave_secret) ? 'sandbox' : 'live';
        $paymentGatewayCredentials->save();
    }

    private function paymentGatewayEnumChange()
    {
        $credentials = PaymentGatewayCredentials::withoutGlobalScope(CompanyScope::class);
        $razorpayCredentials = $credentials->clone()->where('razorpay_status', 'deactive')->orWhereNull('razorpay_status')->orWhere('razorpay_status', '');
        $razorpayCredentials->update(['razorpay_status' => null]);
        \DB::statement('ALTER TABLE payment_gateway_credentials MODIFY razorpay_status ENUM("active", "inactive") DEFAULT "inactive"');
        $razorpayCredentials->update(['razorpay_status' => 'inactive']);

        $paystackCredentials = $credentials->clone()->where('paystack_status', 'inactive')->orWhereNull('paystack_status')->orWhere('paystack_status', '');
        $paystackCredentials->update(['paystack_status' => null]);
        \DB::statement('ALTER TABLE payment_gateway_credentials MODIFY paystack_status ENUM("active", "deactive") DEFAULT "deactive"');
        $paystackCredentials->update(['paystack_status' => 'deactive']);

        $mollieCredentials = $credentials->clone()->where('mollie_status', 'inactive')->orWhereNull('mollie_status')->orWhere('mollie_status', '');
        $mollieCredentials->update(['mollie_status' => null]);
        \DB::statement('ALTER TABLE payment_gateway_credentials MODIFY mollie_status ENUM("active", "deactive") DEFAULT "deactive"');
        $mollieCredentials->update(['mollie_status' => 'deactive']);

        $payfastCredentials = $credentials->clone()->where('payfast_status', 'inactive')->orWhereNull('payfast_status')->orWhere('payfast_status', '');
        $payfastCredentials->update(['payfast_status' => null]);
        \DB::statement('ALTER TABLE payment_gateway_credentials MODIFY payfast_status ENUM("active", "deactive") DEFAULT "deactive"');
        $payfastCredentials->update(['payfast_status' => 'deactive']);

        $authorizeCredentials = $credentials->clone()->where('authorize_status', 'inactive')->orWhereNull('authorize_status')->orWhere('authorize_status', '');
        $authorizeCredentials->update(['authorize_status' => null]);
        \DB::statement('ALTER TABLE payment_gateway_credentials MODIFY authorize_status ENUM("active", "deactive") DEFAULT "deactive"');
        $authorizeCredentials->update(['authorize_status' => 'deactive']);


        $globalCredentials = GlobalPaymentGatewayCredentials::withoutGlobalScope(CompanyScope::class)->where('razorpay_status', 'deactive')->orWhereNull('razorpay_status')->orWhere('razorpay_status', '');
        $globalCredentials->update(['razorpay_status' => null]);
        \DB::statement('ALTER TABLE global_payment_gateway_credentials MODIFY razorpay_status ENUM("active", "inactive") DEFAULT "inactive"');
        $globalCredentials->update(['razorpay_status' => 'inactive']);

    }

    public function languageFixFront()
    {
        $tables = [
            'features',
            'footer_menu',
            'front_clients',
            'front_features',
            'front_faqs',
            'front_features',
            'front_menu_buttons',
            'tr_front_details'
        ];

        $englishLanguage = LanguageSetting::where('language_code', 'en')->first();
        $langId = $englishLanguage?->id;

        foreach ($tables as $table) {
            // Delete the 'en' language from data for newly created and change the existing to this one if null is found
            if(DB::table($table)->whereNull('language_setting_id')->count() > 0) {
                DB::table($table)->where('language_setting_id', $langId)->delete();
            }

            DB::table($table)
                ->whereNull('language_setting_id')
                ->update(['language_setting_id' => $langId]);
        }
    }

    private function fixCustomLeadForm()
    {
        DB::statement("UPDATE `lead_custom_forms` SET `required`='0' where field_name='email'");
    }

    private function storageSettingFix()
    {
        DB::statement("UPDATE `file_storage_settings` SET `filesystem`='aws_s3' where filesystem='aws'");
    }

    private function packageFix()
    {
        DB::statement("UPDATE packages SET module_in_package=REPLACE( module_in_package, 'ticket support', 'tickets' )");
        DB::statement("UPDATE packages SET module_in_package=REPLACE( module_in_package, 'Zoom', 'zoom' )");
    }

    private function logoUpdate()
    {
        DB::statement('UPDATE `companies` SET light_logo=logo');
        DB::statement('UPDATE `global_settings` SET light_logo=logo');

        if(!Schema::hasColumns('contracts', ['contract_note', 'cell', 'office'])) {
            DB::statement('UPDATE `contracts` SET contract_note=description,cell=mobile,office=office_phone');
        }

        SlackSetting::whereNotNull('slack_webhook')->update(['status' => 'active']);

        $globalSetting = GlobalSetting::first();

        if ($globalSetting) {
            DB::statement("UPDATE `companies` SET `favicon`='$globalSetting->favicon'");
        }
    }

    private function hashGenerate()
    {
        $tableNames = ['estimates', 'invoices', 'projects', 'leads', 'proposals', 'contracts'];

        foreach ($tableNames as $tableName) {
            DB::statement("UPDATE `$tableName` SET `hash` = MD5(CONCAT(id, created_at))");
        }
    }

    private function customFieldGroup($company)
    {

        $fields = CustomFieldGroup::ALL_FIELDS;

        foreach ($fields as $field) {
            CustomFieldGroup::updateOrCreate(
                ['name' => $field['name'], 'company_id' => $company->id],
                ['model' => $field['model']]
            );
        }

    }

    public function seedPermission($companyId)
    {

        $employeePermissionSeeder = new EmployeePermissionSeeder();
        $employeePermissionSeeder->insertUserRolePermission($companyId);
    }

    private function languageSettings()
    {
        $languages = LanguageSetting::LANGUAGES;

        foreach ($languages as $language) {
            LanguageSetting::updateOrCreate(['language_code' => $language['language_code']], $language);
        }
    }

    private function customFieldFix()
    {
        $fields = CustomField::get();

        foreach ($fields as $field) {
            $group = CustomFieldGroup::find($field->custom_field_group_id);
            if ($group) {
                $field->company_id = $group->company_id;
                $field->saveQuietly();
            }
        }

        $groupCompany = CustomFieldGroup::where('name', 'Company')->first();

        if ($groupCompany) {
            $groupCompany->model = Company::CUSTOM_FIELD_MODEL;
            $groupCompany->saveQuietly();
        }

        $groupNullTimeLog = CustomFieldGroup::whereNull('company_id')->where('name', 'Time Log')->first();
        $groupNullTimeLog?->delete();
    }

    private function flushAll()
    {
        Artisan::call('optimize:clear');
        Artisan::call('view:clear');
        cache()->flush();
        session()->flush();
        Auth::logout();
    }

}
