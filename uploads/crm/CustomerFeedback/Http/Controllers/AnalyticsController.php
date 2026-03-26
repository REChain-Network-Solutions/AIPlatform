<?php

namespace Modules\CustomerFeedback\Http\Controllers;

use Carbon\Carbon;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;
use App\Http\Controllers\AccountBaseController;
use Modules\CustomerFeedback\Entities\FeedbackTicket;
use Modules\CustomerFeedback\Entities\NpsResponse;
use Modules\CustomerFeedback\Entities\CsatResponse;

class AnalyticsController extends AccountBaseController
{
    public function __construct()
    {
        parent::__construct();
        $this->pageTitle = 'customer-feedback::modules.analytics';
        $this->middleware(function ($request, $next) {
            abort_403(!in_array('customer-feedback', $this->user->modules));
            return $next($request);
        });
    }

    /**
     * Main analytics dashboard
     */
    public function dashboard()
    {
        $startDate = request()->start_date ? Carbon::parse(request()->start_date) : now()->subDays(30);
        $endDate = request()->end_date ? Carbon::parse(request()->end_date) : now();

        // Key metrics
        $this->totalTickets = FeedbackTicket::whereBetween('created_at', [$startDate, $endDate])
            ->count();
        $this->openTickets = FeedbackTicket::whereBetween('created_at', [$startDate, $endDate])
            ->unresolved()
            ->count();
        $this->averageResolutionTime = $this->getAverageResolutionTime($startDate, $endDate);
        $this->satisfactionScore = $this->getAverageSatisfaction($startDate, $endDate);

        // Charts data
        $this->ticketsOverTime = $this->getTicketsOverTime($startDate, $endDate);
        $this->statusBreakdown = $this->getStatusBreakdown();
        $this->priorityBreakdown = $this->getPriorityBreakdown();
        $this->typeBreakdown = $this->getFeedbackTypeBreakdown();
        $this->topAgents = $this->getTopAgents(5);
        $this->npsChart = $this->getNpsData($startDate, $endDate);
        $this->csatChart = $this->getCsatData($startDate, $endDate);

        return view('customer-feedback::analytics.dashboard', $this->data);
    }

    /**
     * NPS analytics
     */
    public function nps()
    {
        $startDate = request()->start_date ? Carbon::parse(request()->start_date) : now()->subDays(90);
        $endDate = request()->end_date ? Carbon::parse(request()->end_date) : now();

        $responses = NpsResponse::whereBetween('created_at', [$startDate, $endDate])->get();

        $total = $responses->count();
        if ($total > 0) {
            $this->promoters = $responses->where('score', '>=', 9)->count();
            $this->passives = $responses->where('score', '>=', 7)->where('score', '<=', 8)->count();
            $this->detractors = $responses->where('score', '<=', 6)->count();
            $this->npsScore = (($this->promoters - $this->detractors) / $total) * 100;
        }

        $this->scoresOverTime = $this->getNpsScoresOverTime($startDate, $endDate);
        $this->scoreDistribution = $this->getNpsScoreDistribution();
        $this->feedback = NpsResponse::where('feedback', '!=', null)
            ->whereBetween('created_at', [$startDate, $endDate])
            ->orderBy('created_at', 'desc')
            ->limit(20)
            ->get();

        return view('customer-feedback::analytics.nps', $this->data);
    }

    /**
     * CSAT analytics
     */
    public function csat()
    {
        $startDate = request()->start_date ? Carbon::parse(request()->start_date) : now()->subDays(90);
        $endDate = request()->end_date ? Carbon::parse(request()->end_date) : now();

        $responses = CsatResponse::whereBetween('created_at', [$startDate, $endDate])->get();

        $this->averageScore = $responses->avg('score');
        $this->responses = $responses->count();
        $this->satisfiedCount = $responses->where('score', '>=', 4)->count();
        $this->satisfactionRate = $this->responses > 0
            ? round(($this->satisfiedCount / $this->responses) * 100, 2)
            : 0;

        $this->scoresOverTime = $this->getCsatScoresOverTime($startDate, $endDate);
        $this->scoreDistribution = $this->getCsatScoreDistribution();

        return view('customer-feedback::analytics.csat', $this->data);
    }

    /**
     * Get average resolution time in hours
     */
    private function getAverageResolutionTime($startDate, $endDate)
    {
        $resolved = FeedbackTicket::whereNotNull('resolved_at')
            ->whereBetween('created_at', [$startDate, $endDate])
            ->get();

        if ($resolved->isEmpty()) {
            return 0;
        }

        $totalHours = $resolved->sum(function ($ticket) {
            return $ticket->resolved_at->diffInHours($ticket->created_at);
        });

        return round($totalHours / $resolved->count(), 2);
    }

    /**
     * Get average satisfaction (NPS + CSAT combined)
     */
    private function getAverageSatisfaction($startDate, $endDate)
    {
        $npsAvg = NpsResponse::whereBetween('created_at', [$startDate, $endDate])
            ->avg('score') ?? 0;

        $csatAvg = CsatResponse::whereBetween('created_at', [$startDate, $endDate])
            ->avg('score') ?? 0;

        return round(($npsAvg + $csatAvg) / 2, 2);
    }

    /**
     * Get tickets over time (daily)
     */
    private function getTicketsOverTime($startDate, $endDate)
    {
        return FeedbackTicket::selectRaw('DATE(created_at) as date, COUNT(*) as count')
            ->whereBetween('created_at', [$startDate, $endDate])
            ->groupBy('date')
            ->get()
            ->map(fn ($item) => [
                'date' => $item->date,
                'count' => $item->count,
            ]);
    }

    /**
     * Get status breakdown
     */
    private function getStatusBreakdown()
    {
        return FeedbackTicket::selectRaw('status, COUNT(*) as count')
            ->groupBy('status')
            ->get()
            ->map(fn ($item) => [
                'status' => $item->status,
                'count' => $item->count,
            ]);
    }

    /**
     * Get priority breakdown
     */
    private function getPriorityBreakdown()
    {
        return FeedbackTicket::selectRaw('priority, COUNT(*) as count')
            ->groupBy('priority')
            ->get()
            ->map(fn ($item) => [
                'priority' => $item->priority,
                'count' => $item->count,
            ]);
    }

    /**
     * Get feedback type breakdown
     */
    private function getFeedbackTypeBreakdown()
    {
        return FeedbackTicket::selectRaw('feedback_type, COUNT(*) as count')
            ->groupBy('feedback_type')
            ->get()
            ->map(fn ($item) => [
                'type' => $item->feedback_type,
                'count' => $item->count,
            ]);
    }

    /**
     * Get top performing agents
     */
    private function getTopAgents($limit = 5)
    {
        return FeedbackTicket::whereNotNull('agent_id')
            ->selectRaw('agent_id, COUNT(*) as handled, SUM(CASE WHEN status IN ("resolved", "closed") THEN 1 ELSE 0 END) as resolved')
            ->groupBy('agent_id')
            ->orderBy('resolved', 'desc')
            ->limit($limit)
            ->with('agent')
            ->get();
    }

    /**
     * Get NPS data for charts
     */
    private function getNpsData($startDate, $endDate)
    {
        $responses = NpsResponse::whereBetween('created_at', [$startDate, $endDate])->get();
        $total = $responses->count();

        return [
            'promoters' => $total > 0 ? ($responses->where('score', '>=', 9)->count() / $total) * 100 : 0,
            'passives' => $total > 0 ? ($responses->where('score', '>=', 7)->where('score', '<=', 8)->count() / $total) * 100 : 0,
            'detractors' => $total > 0 ? ($responses->where('score', '<=', 6)->count() / $total) * 100 : 0,
        ];
    }

    /**
     * Get CSAT data for charts
     */
    private function getCsatData($startDate, $endDate)
    {
        $responses = CsatResponse::whereBetween('created_at', [$startDate, $endDate])->get();
        $total = $responses->count();

        return [
            'satisfied' => $total > 0 ? ($responses->where('score', '>=', 4)->count() / $total) * 100 : 0,
            'neutral' => $total > 0 ? ($responses->where('score', 3)->count() / $total) * 100 : 0,
            'dissatisfied' => $total > 0 ? ($responses->where('score', '<=', 2)->count() / $total) * 100 : 0,
        ];
    }

    /**
     * Get NPS scores over time
     */
    private function getNpsScoresOverTime($startDate, $endDate)
    {
        return NpsResponse::selectRaw('DATE(created_at) as date, AVG(score) as avg_score, COUNT(*) as count')
            ->whereBetween('created_at', [$startDate, $endDate])
            ->groupBy('date')
            ->get();
    }

    /**
     * Get NPS score distribution
     */
    private function getNpsScoreDistribution()
    {
        return NpsResponse::selectRaw('score, COUNT(*) as count')
            ->groupBy('score')
            ->orderBy('score')
            ->get()
            ->map(fn ($item) => [
                'score' => $item->score,
                'count' => $item->count,
            ]);
    }

    /**
     * Get CSAT scores over time
     */
    private function getCsatScoresOverTime($startDate, $endDate)
    {
        return CsatResponse::selectRaw('DATE(created_at) as date, AVG(score) as avg_score, COUNT(*) as count')
            ->whereBetween('created_at', [$startDate, $endDate])
            ->groupBy('date')
            ->get();
    }

    /**
     * Get CSAT score distribution
     */
    private function getCsatScoreDistribution()
    {
        return CsatResponse::selectRaw('score, COUNT(*) as count')
            ->groupBy('score')
            ->orderBy('score')
            ->get()
            ->map(fn ($item) => [
                'score' => $item->score,
                'count' => $item->count,
            ]);
    }
}
