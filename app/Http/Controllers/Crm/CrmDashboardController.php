<?php

namespace App\Http\Controllers\Crm;

use App\Http\Controllers\Controller;
use App\Models\Crm\CrmActivity;
use App\Models\Crm\CrmCompany;
use App\Models\Crm\CrmContact;
use App\Models\Crm\CrmDeal;
use App\Models\Crm\CrmPipeline;
use Illuminate\Support\Facades\Auth;
use Illuminate\View\View;

class CrmDashboardController extends Controller
{
    public function index(): View
    {
        $userId = Auth::id();

        $stats = [
            'contacts'      => CrmContact::forUser($userId)->count(),
            'companies'     => CrmCompany::forUser($userId)->count(),
            'open_deals'    => CrmDeal::forUser($userId)->open()->count(),
            'pipeline_value' => CrmDeal::forUser($userId)->open()->sum('value'),
            'won_this_month' => CrmDeal::forUser($userId)
                ->where('status', 'won')
                ->whereMonth('closed_at', now()->month)
                ->sum('value'),
            'overdue_tasks' => CrmActivity::forUser($userId)->overdue()->count(),
        ];

        $recentContacts = CrmContact::forUser($userId)->with('company')->latest()->limit(5)->get();

        $recentDeals = CrmDeal::forUser($userId)->open()
            ->with(['stage', 'contact'])
            ->orderByDesc('value')
            ->limit(5)
            ->get();

        $upcomingActivities = CrmActivity::forUser($userId)
            ->pending()
            ->with(['contact', 'deal'])
            ->orderBy('due_at')
            ->limit(8)
            ->get();

        $pipelines = CrmPipeline::forUser($userId)->with('stages.deals')->get();

        return view('crm.dashboard', compact(
            'stats',
            'recentContacts',
            'recentDeals',
            'upcomingActivities',
            'pipelines'
        ));
    }
}
