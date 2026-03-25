<?php

namespace Modules\PMCore\app\Http\Controllers;

use App\ApiClasses\Error;
use App\ApiClasses\Success;
use App\Http\Controllers\Controller;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Facades\Validator;
use Illuminate\Support\Str;
use Modules\PMCore\app\Models\ProjectStatus as ProjectStatusModel;

class ProjectStatusController extends Controller
{
    /**
     * Display a listing of project statuses.
     */
    public function index()
    {
        $statuses = ProjectStatusModel::orderBy('sort_order')->get();

        return view('pmcore::project-statuses.index', compact('statuses'));
    }

    /**
     * Get project status data for DataTables.
     */
    public function getDataAjax(Request $request)
    {
        $query = ProjectStatusModel::withCount('projects')
            ->orderBy('sort_order');

        return datatables($query)
            ->editColumn('sort_order', function ($status) {
                return '<i class="bx bx-menu drag-handle" style="cursor: move; font-size: 1.25rem;" title="'.__('Drag to reorder').'"></i>';
            })
            ->addColumn('actions', function ($status) {
                $actions = [
                    [
                        'label' => __('Edit'),
                        'icon' => 'bx bx-edit',
                        'onclick' => "editStatus({$status->id})",
                    ],
                ];

                // Add set as default option if not already default
                if (! $status->is_default) {
                    $actions[] = [
                        'label' => __('Set as Default'),
                        'icon' => 'bx bx-star',
                        'onclick' => "setDefaultStatus({$status->id})",
                    ];
                }

                // Add toggle active/inactive
                $actions[] = [
                    'label' => $status->is_active ? __('Deactivate') : __('Activate'),
                    'icon' => $status->is_active ? 'bx bx-pause' : 'bx bx-play',
                    'onclick' => "toggleStatus({$status->id})",
                    'class' => $status->is_active ? 'text-warning' : 'text-success',
                ];

                // Add delete option if not default and no projects
                if (! $status->is_default && $status->projects_count == 0) {
                    $actions[] = [
                        'divider' => true,
                    ];
                    $actions[] = [
                        'label' => __('Delete'),
                        'icon' => 'bx bx-trash',
                        'onclick' => "deleteStatus({$status->id})",
                        'class' => 'text-danger',
                    ];
                }

                return view('components.datatable-actions', [
                    'id' => $status->id,
                    'actions' => $actions,
                ])->render();
            })
            ->editColumn('name', function ($status) {
                return '<span class="badge" style="background-color: '.$status->color.'; color: white;">'.$status->name.'</span>';
            })
            ->addColumn('projects_count', function ($status) {
                return '<span class="badge bg-info">'.$status->projects_count.'</span>';
            })
            ->editColumn('is_active', function ($status) {
                return $status->is_active ?
                    '<span class="badge bg-success">'.__('Active').'</span>' :
                    '<span class="badge bg-secondary">'.__('Inactive').'</span>';
            })
            ->editColumn('is_default', function ($status) {
                return $status->is_default ?
                    '<span class="badge bg-primary">'.__('Yes').'</span>' :
                    '<span class="badge bg-secondary">'.__('No').'</span>';
            })
            ->editColumn('is_completed', function ($status) {
                return $status->is_completed ?
                    '<span class="badge bg-success">'.__('Yes').'</span>' :
                    '<span class="badge bg-secondary">'.__('No').'</span>';
            })
            ->rawColumns(['sort_order', 'actions', 'name', 'projects_count', 'is_active', 'is_default', 'is_completed'])
            ->make(true);
    }

    /**
     * Store a newly created project status.
     */
    public function store(Request $request)
    {
        $validator = $this->validateProjectStatus($request);

        if ($validator->fails()) {
            return Error::response([
                'message' => __('Validation failed'),
                'errors' => $validator->errors(),
            ]);
        }

        try {
            $status = null;
            DB::transaction(function () use ($request, &$status) {
                $nextSortOrder = ProjectStatusModel::max('sort_order') + 1;

                $status = ProjectStatusModel::create([
                    'name' => $request->name,
                    'description' => $request->description,
                    'slug' => Str::slug($request->name),
                    'color' => $request->color,
                    'sort_order' => $nextSortOrder,
                    'position' => $nextSortOrder,
                    'is_active' => $request->boolean('is_active', true),
                    'is_default' => $request->boolean('is_default', false),
                    'is_completed' => $request->boolean('is_completed', false),
                ]);

                // If this is set as default, remove default from others
                if ($request->boolean('is_default')) {
                    ProjectStatusModel::where('is_default', true)
                        ->where('id', '!=', $status->id)
                        ->update(['is_default' => false]);
                }
            });

            return Success::response([
                'message' => __('Project status created successfully!'),
            ]);

        } catch (\Exception $e) {
            Log::error('Failed to create project status: '.$e->getMessage());

            return Error::response(__('Failed to create project status. Please try again.'));
        }
    }

    /**
     * Display the specified project status.
     */
    public function show(ProjectStatusModel $projectStatus)
    {
        if (request()->ajax()) {
            return Success::response([
                'status' => [
                    'id' => $projectStatus->id,
                    'name' => $projectStatus->name,
                    'description' => $projectStatus->description,
                    'color' => $projectStatus->color,
                    'sort_order' => $projectStatus->sort_order,
                    'is_active' => $projectStatus->is_active,
                    'is_default' => $projectStatus->is_default,
                    'is_completed' => $projectStatus->is_completed,
                ],
            ]);
        }

        return view('pmcore::project-statuses.show', compact('projectStatus'));
    }

    /**
     * Update the specified project status.
     */
    public function update(Request $request, ProjectStatusModel $projectStatus)
    {
        $validator = $this->validateProjectStatus($request, $projectStatus->id);

        if ($validator->fails()) {
            return Error::response([
                'message' => __('Validation failed'),
                'errors' => $validator->errors(),
            ]);
        }

        try {
            DB::transaction(function () use ($request, $projectStatus) {
                $projectStatus->update([
                    'name' => $request->name,
                    'description' => $request->description,
                    'color' => $request->color,
                    'is_active' => $request->boolean('is_active'),
                    'is_default' => $request->boolean('is_default'),
                    'is_completed' => $request->boolean('is_completed'),
                ]);

                // If this is set as default, remove default from others
                if ($request->boolean('is_default')) {
                    ProjectStatusModel::where('is_default', true)
                        ->where('id', '!=', $projectStatus->id)
                        ->update(['is_default' => false]);
                }
            });

            return Success::response([
                'message' => __('Project status updated successfully!'),
            ]);

        } catch (\Exception $e) {
            Log::error('Failed to update project status: '.$e->getMessage());

            return Error::response(__('Failed to update project status. Please try again.'));
        }
    }

    /**
     * Remove the specified project status.
     */
    public function destroy(ProjectStatusModel $projectStatus)
    {
        try {
            // Check if status is being used by any projects
            $projectsCount = $projectStatus->projects()->count();
            if ($projectsCount > 0) {
                return Error::response(
                    __('Cannot delete this status as it is being used by :count projects.', ['count' => $projectsCount])
                );
            }

            // Prevent deletion of default status
            if ($projectStatus->is_default) {
                return Error::response(__('Cannot delete the default project status.'));
            }

            $projectStatus->delete();

            return Success::response([
                'message' => __('Project status deleted successfully!'),
            ]);

        } catch (\Exception $e) {
            Log::error('Failed to delete project status: '.$e->getMessage());

            return Error::response(__('Failed to delete project status. Please try again.'));
        }
    }

    /**
     * Update the sort order of project statuses.
     */
    public function updateSortOrder(Request $request)
    {
        $validator = Validator::make($request->all(), [
            'status_ids' => 'required|array',
            'status_ids.*' => 'exists:project_statuses,id',
        ]);

        if ($validator->fails()) {
            return Error::response([
                'message' => __('Invalid data provided'),
                'errors' => $validator->errors(),
            ]);
        }

        try {
            DB::transaction(function () use ($request) {
                foreach ($request->status_ids as $index => $statusId) {
                    ProjectStatusModel::where('id', $statusId)
                        ->update(['sort_order' => $index + 1]);
                }
            });

            return Success::response([
                'message' => __('Sort order updated successfully!'),
            ]);

        } catch (\Exception $e) {
            Log::error('Failed to update sort order: '.$e->getMessage());

            return Error::response(__('Failed to update sort order. Please try again.'));
        }
    }

    /**
     * Toggle the active status of a project status.
     */
    public function toggleActive(ProjectStatusModel $projectStatus)
    {
        try {
            $projectStatus->update([
                'is_active' => ! $projectStatus->is_active,
            ]);

            return Success::response([
                'message' => __('Project status updated successfully!'),
                'is_active' => $projectStatus->is_active,
            ]);

        } catch (\Exception $e) {
            Log::error('Failed to toggle project status: '.$e->getMessage());

            return Error::response(__('Failed to update project status. Please try again.'));
        }
    }

    /**
     * Set a status as default.
     */
    public function setAsDefault(Request $request)
    {
        $validator = Validator::make($request->all(), [
            'id' => 'required|exists:project_statuses,id',
        ]);

        if ($validator->fails()) {
            return Error::response([
                'message' => __('Invalid status selected'),
                'errors' => $validator->errors(),
            ]);
        }

        try {
            DB::transaction(function () use ($request) {
                // Remove default from all statuses
                ProjectStatusModel::where('is_default', true)->update(['is_default' => false]);

                // Set the selected status as default
                ProjectStatusModel::find($request->id)->update(['is_default' => true]);
            });

            return Success::response([
                'message' => __('Default status updated successfully!'),
            ]);

        } catch (\Exception $e) {
            Log::error('Failed to set default status: '.$e->getMessage());

            return Error::response(__('Failed to update default status. Please try again.'));
        }
    }

    /**
     * Validate project status data.
     */
    protected function validateProjectStatus(Request $request, $statusId = null)
    {
        $rules = [
            'name' => 'required|string|max:255|unique:project_statuses,name,'.$statusId,
            'description' => 'nullable|string|max:1000',
            'color' => 'required|string|regex:/^#[a-fA-F0-9]{6}$/',
            'is_active' => 'boolean',
            'is_default' => 'boolean',
            'is_completed' => 'boolean',
        ];

        return Validator::make($request->all(), $rules);
    }
}
