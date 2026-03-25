<?php

namespace Workdo\Pos\Listeners;

use App\Events\CompanyMenuEvent;

class CompanyMenuListener
{
    /**
     * Handle the event.
     */
    public function handle(CompanyMenuEvent $event): void
    {
        $module = 'Pos';
        $menu = $event->menu;
        $menu->add([
            'category' => 'General',
            'title' => __('POS Dashboard'),
            'icon' => '',
            'name' => 'pos-dashboard',
            'parent' => 'dashboard',
            'order' => 40,
            'ignore_if' => [],
            'depend_on' => [],
            'route' => 'pos.dashboard',
            'module' => $module,
            'permission' => 'pos dashboard manage'
        ]);
        $menu->add([
            'category' => 'Sales',
            'title' => __('POS'),
            'icon' => 'grid-dots',
            'name' => 'pos',
            'parent' => null,
            'order' => 475,
            'ignore_if' => [],
            'depend_on' => [],
            'route' => '',
            'module' => $module,
            'permission' => 'pos manage'
        ]);


        $menu->add([
            'category' => 'Sales',
            'title' => __('Add POS'),
            'icon' => '',
            'name' => 'add-pos',
            'parent' => 'pos',
            'order' => 20,
            'ignore_if' => [],
            'depend_on' => [],
            'route' => 'pos.index',
            'module' => $module,
            'permission' => 'pos add manage'
        ]);
        $menu->add([
            'category' => 'Sales',
            'title' => __('POS Order'),
            'icon' => '',
            'name' => 'pos-order',
            'parent' => 'pos',
            'order' => 25,
            'ignore_if' => [],
            'depend_on' => [],
            'route' => 'pos.report',
            'module' => $module,
            'permission' => 'pos order manage'
        ]);
        $menu->add([
            'category' => 'Sales',
            'title' => __('Print Barcode'),
            'icon' => '',
            'name' => 'pos-print-barcode',
            'parent' => 'pos',
            'order' => 30,
            'ignore_if' => [],
            'depend_on' => [],
            'route' => 'pos.barcode',
            'module' => $module,
            'permission' => 'print_barcode manage'
        ]);
        $menu->add([
            'category' => 'Sales',
            'title' => __('Report'),
            'icon' => '',
            'name' => 'pos-reports',
            'parent' => 'pos',
            'order' => 35,
            'ignore_if' => [],
            'depend_on' => [],
            'route' => '',
            'module' => $module,
            'permission' => 'report pos'
        ]);


        $menu->add([
            'category' => 'Sales',
            'title' => __('Pos Daily/Monthly Report'),
            'icon' => '',
            'name' => 'pos-report',
            'parent' => 'pos-reports',
            'order' => 20,
            'ignore_if' => [],
            'depend_on' => [],
            'route' => 'report.daily.pos',
            'module' => $module,
            'permission' => 'report pos'
        ]);
        $menu->add([
            'category' => 'Sales',
            'title' => __('Pos VS Purchase Report'),
            'icon' => '',
            'name' => 'pos-vs-purchase-report',
            'parent' => 'pos-reports',
            'order' => 25,
            'ignore_if' => [],
            'depend_on' => [],
            'route' => 'report.pos.vs.purchase',
            'module' => $module,
            'permission' => 'report pos vs expense'
        ]);
    }
}
