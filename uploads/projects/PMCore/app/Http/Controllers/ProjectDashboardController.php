<?php

namespace Modules\PMCore\app\Http\Controllers;

use App\Http\Controllers\Controller;
use Modules\PMCore\app\Enums\ProjectStatus;
use Modules\PMCore\app\Models\Project;
use Modules\PMCore\app\Services\PMIntegrationService;

class ProjectDashboardController extends Controller
{
    protected $integrationService;

    public function __construct(PMIntegrationService $integrationService)
    {
        $this->integrationService = $integrationService;
    }

    /**
     * Display the project dashboard.
     */
    public function index()
    {
        $stats = $this->getDashboardStats();
        $chartData = $this->getChartData();
        $recentProjects = $this->getRecentProjects();
        $overdueProjects = $this->getOverdueProjects();

        return view('pmcore::dashboard.index', compact(
            'stats',
            'chartData',
            'recentProjects',
            'overdueProjects'
        ));
    }

    /**
     * Get dashboard statistics.
     */
    protected function getDashboardStats()
    {
        $totalProjects = Project::count();
        $activeProjects = Project::where('status', ProjectStatus::IN_PROGRESS)->count();
        $completedProjects = Project::where('status', ProjectStatus::COMPLETED)->count();
        $overdueProjects = Project::where('end_date', '<', now())
            ->whereNotIn('status', [ProjectStatus::COMPLETED, ProjectStatus::CANCELLED])
            ->count();

        $completionRate = $totalProjects > 0 ? round(($completedProjects / $totalProjects) * 100, 2) : 0;

        return [
            'total_projects' => $totalProjects,
            'active_projects' => $activeProjects,
            'completed_projects' => $completedProjects,
            'overdue_projects' => $overdueProjects,
            'completion_rate' => $completionRate,
            'total_budget' => Project::sum('budget'),
            'active_budget' => Project::where('status', ProjectStatus::IN_PROGRESS)->sum('budget'),
        ];
    }

    /**
     * Get chart data for dashboard.
     */
    protected function getChartData()
    {
        // Project status distribution
        $statusData = [
            'labels' => [],
            'data' => [],
            'colors' => [],
        ];

        foreach (ProjectStatus::cases() as $status) {
            $count = Project::where('status', $status)->count();
            if ($count > 0) {
                $statusData['labels'][] = $status->label();
                $statusData['data'][] = $count;
                $statusData['colors'][] = $status->color();
            }
        }

        // Monthly project creation trend
        $monthlyData = [];
        for ($i = 5; $i >= 0; $i--) {
            $date = now()->subMonths($i);
            $count = Project::whereYear('created_at', $date->year)
                ->whereMonth('created_at', $date->month)
                ->count();
            $monthlyData['labels'][] = $date->format('M Y');
            $monthlyData['data'][] = $count;
        }

        // Project completion trend
        $completionData = [];
        for ($i = 5; $i >= 0; $i--) {
            $date = now()->subMonths($i);
            $completed = Project::where('status', ProjectStatus::COMPLETED)
                ->whereYear('updated_at', $date->year)
                ->whereMonth('updated_at', $date->month)
                ->count();
            $completionData['labels'][] = $date->format('M Y');
            $completionData['data'][] = $completed;
        }

        return [
            'status_distribution' => $statusData,
            'monthly_creation' => $monthlyData,
            'completion_trend' => $completionData,
        ];
    }

    /**
     * Get recent projects.
     */
    protected function getRecentProjects()
    {
        return Project::with(['client', 'projectManager'])
            ->orderBy('created_at', 'desc')
            ->limit(10)
            ->get();
    }

    /**
     * Get overdue projects.
     */
    protected function getOverdueProjects()
    {
        return Project::with(['client', 'projectManager'])
            ->where('end_date', '<', now())
            ->whereNotIn('status', [ProjectStatus::COMPLETED, ProjectStatus::CANCELLED])
            ->orderBy('end_date', 'asc')
            ->limit(10)
            ->get();
    }
}
