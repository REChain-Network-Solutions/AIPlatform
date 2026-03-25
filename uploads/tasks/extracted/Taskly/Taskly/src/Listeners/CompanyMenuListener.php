<?php

namespace Workdo\Taskly\Listeners;

use App\Events\CompanyMenuEvent;

class CompanyMenuListener
{
    /**
     * Handle the event.
     */
    public function handle(CompanyMenuEvent $event): void
    {
        $module = 'Taskly';
        $menu = $event->menu;
        $menu->add([
            'category' => 'General',
            'title' => __('Project Dashboard'),
            'icon' => '',
            'name' => 'taskly-dashboards',
            'parent' => 'dashboard',
            'order' => 10,
            'ignore_if' => [],
            'depend_on' => [],
            'route' => 'taskly.dashboard',
            'module' => $module,
            'permission' => 'taskly dashboard manage'
        ]);
        $menu->add([
            'category' => 'Finance',
            'title' => __('Projects'),
            'icon' => 'square-check',
            'name' => 'projects',
            'parent' => null,
            'order' => 300,
            'ignore_if' => [],
            'depend_on' => [],
            'route' => '',
            'module' => $module,
            'permission' => 'project manage'
        ]);
        $menu->add([
            'category' => 'Finance',
            'title' => __('Project'),
            'icon' => '',
            'name' => 'project',
            'parent' => 'projects',
            'order' => 10,
            'ignore_if' => [],
            'depend_on' => [],
            'route' => 'projects.index',
            'module' => $module,
            'permission' => 'project manage'
        ]);
        $menu->add([
            'category' => 'Finance',
            'title' => __('Project Report'),
            'icon' => '',
            'name' => 'project-report',
            'parent' => 'projects',
            'order' => 20,
            'ignore_if' => [],
            'depend_on' => [],
            'route' => 'project_report.index',
            'module' => $module,
            'permission' => 'project report manage'
        ]);
        $menu->add([
            'category' => 'Finance',
            'title' => __('System Setup'),
            'icon' => '',
            'name' => 'system-setup',
            'parent' => 'projects',
            'order' => 30,
            'ignore_if' => [],
            'depend_on' => [],
            'route' => 'stages.index',
            'module' => $module,
            'permission' => 'taskly setup manage'
        ]);
    }
}
