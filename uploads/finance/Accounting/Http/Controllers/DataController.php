<?php

namespace Modules\Accounting\Http\Controllers;

use App\Utils\ModuleUtil;
use App\Utils\Util;
use Illuminate\Routing\Controller;
use Menu;

class DataController extends Controller
{
    /**
     * Superadmin package permissions
     *
     * @return array
     */
    public function superadmin_package()
    {
        return [
            [
                'name' => 'accounting_module',
                'label' => __('accounting::lang.accounting_module'),
                'default' => false,
            ],
        ];
    }

    /**
     * Adds cms menus
     *
     * @return null
     */
    public function modifyAdminMenu()
    {
        $business_id = session()->get('user.business_id');
        $module_util = new ModuleUtil();

        $is_accounting_enabled = (bool) $module_util->hasThePermissionInSubscription($business_id, 'accounting_module');

        $commonUtil = new Util();
        $is_admin = $commonUtil->is_admin(auth()->user(), $business_id);

        if (auth()->user()->can('accounting.access_accounting_module') && $is_accounting_enabled) {
            Menu::modify(
                'admin-sidebar-menu',
                function ($menu) {
                    $menu->url(action([\Modules\Accounting\Http\Controllers\AccountingController::class, 'dashboard']), __('accounting::lang.accounting'), ['icon' => '<svg aria-hidden="true" class="tw-size-5 tw-shrink-0" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round">
                    <path stroke="none" d="M0 0h24v24H0z" fill="none"></path>
                    <path d="M3 21l18 0"></path>
                    <path d="M3 10l18 0"></path>
                    <path d="M5 6l7 -3l7 3"></path>
                    <path d="M4 10l0 11"></path>
                    <path d="M20 10l0 11"></path>
                    <path d="M8 14l0 3"></path>
                    <path d="M12 14l0 3"></path>
                    <path d="M16 14l0 3"></path>
                  </svg>', 'style' => config('app.env') == 'demo' ? 'background-color: #D483D9;color:white' : '', 'active' => request()->segment(1) == 'accounting'])->order(50);
                }
            );
        }
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
                'value' => 'accounting.access_accounting_module',
                'label' => __('accounting::lang.access_accounting_module'),
                'default' => false,
            ],
            [
                'value' => 'accounting.manage_accounts',
                'label' => __('accounting::lang.manage_accounts'),
                'default' => false,
            ],
            [
                'value' => 'accounting.view_journal',
                'label' => __('accounting::lang.view_journal'),
                'default' => false,
            ],
            [
                'value' => 'accounting.add_journal',
                'label' => __('accounting::lang.add_journal'),
                'default' => false,
            ],
            [
                'value' => 'accounting.edit_journal',
                'label' => __('accounting::lang.edit_journal'),
                'default' => false,
            ],
            [
                'value' => 'accounting.delete_journal',
                'label' => __('accounting::lang.delete_journal'),
                'default' => false,
            ],
            [
                'value' => 'accounting.map_transactions',
                'label' => __('accounting::lang.map_transactions'),
                'default' => false,
            ],
            [
                'value' => 'accounting.view_transfer',
                'label' => __('accounting::lang.view_transfer'),
                'default' => false,
            ],
            [
                'value' => 'accounting.add_transfer',
                'label' => __('accounting::lang.add_transfer'),
                'default' => false,
            ],
            [
                'value' => 'accounting.edit_transfer',
                'label' => __('accounting::lang.edit_transfer'),
                'default' => false,
            ],
            [
                'value' => 'accounting.delete_transfer',
                'label' => __('accounting::lang.delete_transfer'),
                'default' => false,
            ],
            [
                'value' => 'accounting.manage_budget',
                'label' => __('accounting::lang.manage_budget'),
                'default' => false,
            ],
            [
                'value' => 'accounting.view_reports',
                'label' => __('accounting::lang.view_reports'),
                'default' => false,
            ],
        ];
    }
}
