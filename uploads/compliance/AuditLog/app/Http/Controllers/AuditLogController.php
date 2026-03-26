<?php

namespace Modules\AuditLog\app\Http\Controllers;

use App\ApiClasses\Error;
use App\ApiClasses\Success;
use App\Http\Controllers\Controller;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;
use OwenIt\Auditing\Models\Audit;
use Yajra\DataTables\Facades\DataTables;

class AuditLogController extends Controller
{
    /**
     * Display a listing of audit logs.
     */
    public function index()
    {
        $pageData = [
            'title' => __('Audit Logs'),
            'urls' => [
                'datatable' => route('auditlog.datatable'),
                'show' => route('auditlog.show', ':id'),
                'statistics' => route('auditlog.statistics'),
                'filters' => route('auditlog.filters'),
            ],
            'labels' => [
                'confirmDelete' => __('Are you sure you want to delete this audit log?'),
                'success' => __('Success!'),
                'error' => __('Error!'),
            ],
        ];

        return view('auditlog::index', compact('pageData'));
    }

    /**
     * Get audit logs data for DataTables.
     */
    public function indexAjax(Request $request)
    {
        $query = Audit::with(['user', 'auditable']);

        // Apply filters
        if ($request->filled('auditable_type')) {
            $query->where('auditable_type', $request->auditable_type);
        }

        if ($request->filled('event')) {
            $query->where('event', $request->event);
        }

        if ($request->filled('user_id')) {
            $query->where('user_id', $request->user_id);
        }

        if ($request->filled('date_from')) {
            $query->whereDate('created_at', '>=', $request->date_from);
        }

        if ($request->filled('date_to')) {
            $query->whereDate('created_at', '<=', $request->date_to);
        }

        return DataTables::of($query)
            ->addColumn('user', function ($audit) {
                return view('components.datatable-user', ['user' => $audit->user])->render();
            })
            ->addColumn('auditable', function ($audit) {
                $model = class_basename($audit->auditable_type);
                $id = $audit->auditable_id;

                return "<span class='badge bg-label-primary'>{$model}</span> <span class='text-muted'>#{$id}</span>";
            })
            ->addColumn('event', function ($audit) {
                $colors = [
                    'created' => 'success',
                    'updated' => 'info',
                    'deleted' => 'danger',
                    'restored' => 'warning',
                ];
                $color = $colors[$audit->event] ?? 'secondary';

                return "<span class='badge bg-label-{$color}'>".ucfirst($audit->event).'</span>';
            })
            ->addColumn('ip_address', function ($audit) {
                return $audit->ip_address ?: '<span class="text-muted">N/A</span>';
            })
            ->addColumn('created_at', function ($audit) {
                return $audit->created_at->format('Y-m-d H:i:s');
            })
            ->addColumn('actions', function ($audit) {
                return view('components.datatable-actions', [
                    'id' => $audit->id,
                    'actions' => [
                        [
                            'label' => __('View Details'),
                            'icon' => 'bx bx-show',
                            'onclick' => "viewAuditDetails({$audit->id})",
                        ],
                    ],
                ])->render();
            })
            ->rawColumns(['user', 'auditable', 'event', 'ip_address', 'actions'])
            ->make(true);
    }

    /**
     * Display the specified audit log.
     */
    public function show($id)
    {
        try {
            $auditLog = Audit::with(['user', 'auditable'])->findOrFail($id);

            // Format the changes for display
            $oldValues = $auditLog->old_values ?? [];
            $newValues = $auditLog->new_values ?? [];

            $changes = [];
            $allKeys = array_unique(array_merge(array_keys($oldValues), array_keys($newValues)));

            foreach ($allKeys as $key) {
                $changes[] = [
                    'field' => $key,
                    'old' => $oldValues[$key] ?? null,
                    'new' => $newValues[$key] ?? null,
                ];
            }

            $html = view('auditlog::partials.show', compact('auditLog', 'changes'))->render();

            return Success::response([
                'html' => $html,
            ]);
        } catch (\Exception $e) {
            return Error::response($e->getMessage());
        }
    }

    /**
     * Get audit statistics.
     */
    public function statistics(Request $request)
    {
        try {
            $dateFrom = $request->date_from ?? now()->subDays(30)->format('Y-m-d');
            $dateTo = $request->date_to ?? now()->format('Y-m-d');

            // Total audits
            $totalAudits = Audit::whereBetween('created_at', [$dateFrom, $dateTo])->count();

            // Audits by event type
            $auditsByEvent = Audit::whereBetween('created_at', [$dateFrom, $dateTo])
                ->select('event', DB::raw('count(*) as total'))
                ->groupBy('event')
                ->pluck('total', 'event')
                ->toArray();

            // Most active users
            $activeUsers = Audit::whereBetween('created_at', [$dateFrom, $dateTo])
                ->select('user_id', DB::raw('count(*) as total'))
                ->whereNotNull('user_id')
                ->groupBy('user_id')
                ->orderBy('total', 'desc')
                ->limit(5)
                ->get()
                ->map(function ($item) {
                    // Load user relationship if not already loaded
                    if (! $item->relationLoaded('user')) {
                        $item->load('user');
                    }

                    $userName = 'Unknown';
                    if ($item->user) {
                        // Use the full_name accessor from the User model
                        $userName = $item->user->full_name;
                    } else {
                        $userName = "User #{$item->user_id} (Deleted)";
                    }

                    return [
                        'user' => $userName,
                        'total' => $item->total,
                    ];
                });

            // Most audited models
            $auditedModels = Audit::whereBetween('created_at', [$dateFrom, $dateTo])
                ->select('auditable_type', DB::raw('count(*) as total'))
                ->groupBy('auditable_type')
                ->orderBy('total', 'desc')
                ->limit(5)
                ->get()
                ->map(function ($item) {
                    return [
                        'model' => class_basename($item->auditable_type),
                        'total' => $item->total,
                    ];
                });

            // Daily audit trend
            $dailyTrend = Audit::whereBetween('created_at', [$dateFrom, $dateTo])
                ->select(DB::raw('DATE(created_at) as date'), DB::raw('count(*) as total'))
                ->groupBy('date')
                ->orderBy('date')
                ->get()
                ->map(function ($item) {
                    return [
                        'x' => $item->date,
                        'y' => $item->total,
                    ];
                });

            return Success::response([
                'totalAudits' => $totalAudits,
                'auditsByEvent' => $auditsByEvent,
                'activeUsers' => $activeUsers,
                'auditedModels' => $auditedModels,
                'dailyTrend' => $dailyTrend,
            ]);
        } catch (\Exception $e) {
            return Error::response($e->getMessage());
        }
    }

    /**
     * Get available filters data.
     */
    public function filters()
    {
        try {
            // Get unique auditable types
            $auditableTypes = Audit::select('auditable_type')
                ->distinct()
                ->pluck('auditable_type')
                ->map(function ($type) {
                    return [
                        'value' => $type,
                        'label' => class_basename($type),
                    ];
                });

            // Get unique events
            $events = Audit::select('event')
                ->distinct()
                ->pluck('event')
                ->map(function ($event) {
                    return [
                        'value' => $event,
                        'label' => ucfirst($event),
                    ];
                });

            return Success::response([
                'auditableTypes' => $auditableTypes,
                'events' => $events,
            ]);
        } catch (\Exception $e) {
            return Error::response($e->getMessage());
        }
    }
}
