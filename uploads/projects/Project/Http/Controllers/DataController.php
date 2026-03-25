<?php

namespace Modules\Project\Http\Controllers;

use App\TaxRate;
use App\User;
use App\Utils\ModuleUtil;
use App\Utils\Util;
use Illuminate\Routing\Controller;
use Illuminate\Support\Facades\DB;
use Menu;
use Modules\Project\Entities\Project;
use Modules\Project\Entities\ProjectTask;
use Modules\Project\Entities\ProjectTransaction;
use Modules\Project\Utils\ProjectUtil;

class DataController extends Controller
{
    /**
     * Parses notification message from database.
     *
     * @return array
     */
    public function parse_notification($notification)
    {
        $notification_datas = [];
        if ($notification->type == 'Modules\Project\Notifications\NewProjectAssignedNotification') {
            $data = $notification->data;
            $project = Project::with('createdBy')
                ->find($data['project_id']);

            if (!empty($project)) {
                $msg = __(
                    'project::lang.new_project_assgined_notification',
                    [
                        'created_by' => $project->createdBy->user_full_name,
                        'project' => $project->name,
                    ]
                );

                $link = action([\Modules\Project\Http\Controllers\ProjectController::class, 'show'], ['id' => $project->id]);

                $notification_datas = [
                    'msg' => $msg,
                    'icon_class' => 'fas fa fa-check-circle bg-green',
                    'link' => $link,
                    'read_at' => $notification->read_at,
                    'created_at' => $notification->created_at->diffForHumans(),
                ];
            }
        } elseif ($notification->type == 'Modules\Project\Notifications\NewTaskAssignedNotification') {
            $data = $notification->data;
            $task = ProjectTask::with('createdBy')
                ->where('project_id', $data['project_id'])
                ->find($data['project_task_id']);

            if (!empty($task)) {
                $msg = __(
                    'project::lang.new_task_assgined_notification',
                    [
                        'created_by' => $task->createdBy->user_full_name,
                        'subject' => $task->subject,
                        'task_id' => $task->task_id,
                    ]
                );

                $link = action([\Modules\Project\Http\Controllers\ProjectController::class, 'show'], ['id' => $task->project_id]);

                $notification_datas = [
                    'msg' => $msg,
                    'icon_class' => 'fas fa-tasks bg-green',
                    'link' => $link,
                    'read_at' => $notification->read_at,
                    'created_at' => $notification->created_at->diffForHumans(),
                ];
            }
        } elseif ($notification->type == 'Modules\Project\Notifications\NewCommentOnTaskNotification') {
            $data = $notification->data;
            $task = ProjectTask::with('createdBy')
                ->where('project_id', $data['project_id'])
                ->find($data['project_task_id']);

            if (!empty($task)) {
                $user = User::find($data['commented_by']);

                $msg = __(
                    'project::lang.new_comment_on_task_notification',
                    [
                        'commented_by' => $user->user_full_name,
                        'subject' => $task->subject,
                        'task_id' => $task->task_id,
                    ]
                );

                $link = action([\Modules\Project\Http\Controllers\ProjectController::class, 'show'], ['id' => $task->project_id]);

                $notification_datas = [
                    'msg' => $msg,
                    'icon_class' => 'fas fa-comment-dots bg-green',
                    'link' => $link,
                    'read_at' => $notification->read_at,
                    'created_at' => $notification->created_at->diffForHumans(),
                ];
            }
        }

        return $notification_datas;
    }

    /**
     * Adds Project menus
     *
     * @return null
     */
    public function modifyAdminMenu()
    {
        $business_id = session()->get('user.business_id');
        $module_util = new ModuleUtil();
        $commonUtil = new Util();
        $is_admin = $commonUtil->is_admin(auth()->user(), $business_id);

        $is_project_enabled = (bool) $module_util->hasThePermissionInSubscription($business_id, 'project_module');

        if ($is_project_enabled) {
            Menu::modify(
                'admin-sidebar-menu',
                function ($menu) {
                    $menu->url(
                        action([\Modules\Project\Http\Controllers\ProjectController::class, 'index']) . '?project_view=list_view',
                        __('project::lang.project'),
                        ['icon' => '<svg  xmlns="http://www.w3.org/2000/svg"  width="24"  height="24"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round"  class="icon icon-tabler icons-tabler-outline icon-tabler-brand-asana"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M12 7m-3 0a3 3 0 1 0 6 0a3 3 0 1 0 -6 0" /><path d="M17 16m-3 0a3 3 0 1 0 6 0a3 3 0 1 0 -6 0" /><path d="M7 16m-3 0a3 3 0 1 0 6 0a3 3 0 1 0 -6 0" /></svg>', 'active' => request()->segment(1) == 'project' || request()->get('type') == 'project', 'style' => config('app.env') == 'demo' ? 'background-color: #e4186d !important;color:white' : '']
                    )
                        ->order(86);
                }
            );
        }
    }

    /**
     * get gross project from
     * project
     *
     * @param $business_id, $start_date, $end_date,
     *  $location_id
     * @return decimal
     */
    public function grossProfit($params)
    {
        $transaction = ProjectTransaction::where('business_id', $params['business_id'])
            ->where('type', 'sell')
            ->where('sub_type', 'project_invoice')
            ->where('status', 'final');

        if (!empty($params['start_date']) && !empty($params['end_date'])) {
            if ($params['start_date'] == $params['end_date']) {
                $transaction->whereDate('transaction_date', $params['end_date']);
            } else {
                $transaction->whereBetween(DB::raw('transaction_date'), [$params['start_date'], $params['end_date']]);
            }
        }

        $transaction = $transaction->select(DB::raw('SUM(final_total) as gross_profit'))
            ->first();

        $data = [
            'value' => $transaction->gross_profit,
            'label' => __('project::lang.project_invoice'),
        ];

        return $data;
    }

    /**
     * Defines user permissions for the module.
     *
     * @return array
     */
    public function user_permissions()
    {
        return [
            [
                'value' => 'project.create_project',
                'label' => __('project::lang.create_project'),
                'default' => false,
            ],
            [
                'value' => 'project.edit_project',
                'label' => __('project::lang.edit_project'),
                'default' => false,
            ],
            [
                'value' => 'project.delete_project',
                'label' => __('project::lang.delete_project'),
                'default' => false,
            ],
        ];
    }

    /**
     * Superadmin package permissions
     *
     * @return array
     */
    public function superadmin_package()
    {
        return [
            [
                'name' => 'project_module',
                'label' => __('project::lang.project_module'),
                'default' => false,
            ],
        ];
    }

    /**
     * Function to return Project module
     * permission for doc & note if notable type
     * and notable id validates
     *
     * @return array
     */
    public function addDocumentAndNotes($params)
    {
        $permissions = [];

        $notable_type = $params['notable_type'];
        if ($notable_type == 'Modules\Project\Entities\Project') {
            $notable = $notable_type::where('business_id', $params['business_id'])
                ->find($params['notable_id']);

            if (!empty($notable)) {
                //check if user is member/lead/admin
                $commonUtil = new Util();
                $projectUtil = new ProjectUtil();
                $is_admin = $commonUtil->is_admin(auth()->user(), $params['business_id']);
                $is_lead = $projectUtil->isProjectLead(auth()->user()->id, $params['notable_id']);
                $is_member = $projectUtil->isProjectMember(auth()->user()->id, $params['notable_id']);

                //user is member/lead/admin;assign permission
                if (($is_admin || $is_lead)) {
                    $permissions = ['view', 'create', 'delete'];
                } elseif ($is_member) {
                    if (isset($notable->settings['members_crud_note']) && $notable->settings['members_crud_note']) {
                        $permissions = ['view', 'create', 'delete'];
                    } else {
                        $permissions = ['view'];
                    }
                }
            }
        }

        return [
            'Modules\Project\Entities\Project' => [
                'permissions' => $permissions,
            ],
        ];
    }

    /**
     * Function to add project module taxonomies
     *
     * @return array
     */
    public function addTaxonomies()
    {
        $business_id = request()->session()->get('user.business_id');

        $module_util = new ModuleUtil();
        if (!(auth()->user()->can('superadmin') || $module_util->hasThePermissionInSubscription($business_id, 'project_module'))) {
            return ['project' => []];
        }

        return [
            'project' => [
                'taxonomy_label' => __('project::lang.project_category'),
                'heading' => __('project::lang.project_categories'),
                'sub_heading' => __('project::lang.manage_project_categories'),
                'enable_taxonomy_code' => false,
                'enable_sub_taxonomy' => false,
                'navbar' => 'project::layouts.nav',
            ],
        ];
    }

    /**
     * Returns Tab path with required extra data.
     * for tax report view
     *
     * @return array
     */
    public function getTaxReportViewTabs()
    {
        $module_util = new ModuleUtil();
        $business_id = session()->get('user.business_id');
        $is_subscribed_project = (bool) $module_util->hasThePermissionInSubscription($business_id, 'project_module');

        if ($is_subscribed_project) {
            //for multiple tab just add another array of tab details and if js is in common file just include once in any array
            return [
                [
                    'tab_menu_path' => 'project::tax_report.tab_menu',
                    'tab_content_path' => 'project::tax_report.tab_content',
                    'tab_data' => [],
                    'module_js_path' => 'project::tax_report.tax_report_js',
                ],
            ];
        } else {
            return [];
        }
    }

    /**
     * Returns project invoice tax amount
     *
     * for tax report view
     *
     * @return array
     */
    public function getModuleOutputTax($params)
    {
        if (!auth()->user()->can('tax_report.view')) {
            abort(403, 'Unauthorized action.');
        }
        $commonUtil = new Util();
        $start_date = $params['start_date'];
        $end_date = $params['end_date'];
        $business_id = session()->get('user.business_id');
        $taxes = TaxRate::forBusiness($business_id);

        $query = ProjectTransaction::with(['invoiceLines' => function ($q) {
            $q->whereNotNull('pjt_invoice_lines.tax_rate_id');
        }])
            ->where('transactions.business_id', $business_id)
            ->where('transactions.type', 'sell')
            ->where('transactions.sub_type', 'project_invoice')
            ->where('transactions.status', 'final');

        if (!empty($start_date) && !empty($end_date)) {
            $query->whereBetween(DB::raw('date(transaction_date)'), [$start_date, $end_date]);
        }
        $transactions = $query->get();

        $total_tax = 0;
        foreach ($taxes as $key => $tax) {
            foreach ($transactions as $transaction) {
                foreach ($transaction->invoiceLines as $invoiceLine) {
                    if ($invoiceLine->tax_rate_id == $tax['id']) {
                        $row_amount = $invoiceLine->rate * $invoiceLine->quantity;
                        $item_tax = $commonUtil->calc_percentage($row_amount, $tax['amount']);
                        $total_tax += $item_tax * $invoiceLine->quantity;
                    }
                }
            }
        }

        return $total_tax;
    }

    public function CustomDashboardOptions()
    {
        return [
            [
                'name' => 'my_tasks',
                'label' => __('project::lang.my_tasks'),
                'size' => 25,
                'module_name' => __('project::lang.project'),
                'range' => true,
                'html_text' => false,
                'location' => false,
                'show_data' => true,
            ],
        ];
    }

    public function my_tasks($dashboard_detail)
    {

        $business_id = session()->get('user.business_id');
        $user_id = null;

        $start_date = $dashboard_detail->start_date;
        $end_date = $dashboard_detail->end_date;

        if ($dashboard_detail->show_data == 'show_loggedin_user_data') {
            $user_id = session()->get('user.id');
        }

        $task_count = ProjectTask::where('business_id', $business_id)->selectRaw('
                COUNT(CASE WHEN status = "completed" THEN 1 END) AS completed_tasks,
                COUNT(CASE WHEN status = "not_started" THEN 1 END) AS not_started_tasks,
                COUNT(CASE WHEN status = "in_progress" THEN 1 END) AS in_progress_tasks,
                COUNT(CASE WHEN status = "on_hold" THEN 1 END) AS on_hold_tasks,
                COUNT(CASE WHEN status = "cancelled" THEN 1 END) AS cancelled_tasks,
                COUNT(*) AS total_tasks
            ');

        if (!empty($user_id)) {
            $task_count = $task_count->where('created_by', $user_id);
        }

         if (! empty($start_date) && ! empty($end_date)) {
            $task_count = $task_count->whereDate('start_date', '>=', $start_date)
                ->whereDate('start_date', '<=', $end_date);
        }

        $task_count = $task_count->first();

        return [
            'html' => view('project::my_task.custom_dashboard.my_tasks', compact('dashboard_detail', 'task_count')),
        ];
    }
}
