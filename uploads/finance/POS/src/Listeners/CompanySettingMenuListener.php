<?php

namespace Workdo\Pos\Listeners;

use App\Events\CompanySettingMenuEvent;

class CompanySettingMenuListener
{
    /**
     * Handle the event.
     */
    public function handle(CompanySettingMenuEvent $event): void
    {
        $module = 'Pos';
        $menu = $event->menu;
        $menu->add([
            'title' => __('POS Settings'),
            'name' => 'pos-setting',
            'order' => 180,
            'ignore_if' => [],
            'depend_on' => [],
            'route' => '',
            'navigation' => 'pos-sidenav',
            'module' => $module,
            'permission' => 'pos setting manage'
        ]);

        $menu->add([
            'title' => __('Pos Print Settings'),
            'name' => 'pos-print-setting',
            'order' => 200,
            'ignore_if' => [],
            'depend_on' => [],
            'route' => '',
            'navigation' => 'pos-print-sidenav',
            'module' => $module,
            'permission' => 'pos setting manage'
        ]);
    }
}
