<?php

namespace Modules\TasksAssignmentModule\Http\Controllers;

use App\Helper\Reply;
use App\Http\Controllers\AccountBaseController;
use App\Models\TaskAssignment;
use Illuminate\Http\Request;
use App\DataTables\TasksAssignmentDataTable;

class TasksAssignmentModuleController extends AccountBaseController
{
    public function __construct()
    {
        parent::__construct();
        $this->pageTitle = 'app.menu.tasks_assignment';

        $this->middleware(function ($request, $next) {
            abort_if(!in_array('clients', $this->user->modules), 403);
            return $next($request);
        });
    }

    /**
     * Display a listing of the resource.
     */
    public function index(TasksAssignmentDataTable $dataTable)
    {
        if (!request()->ajax()) {
            $this->tasks = TaskAssignment::all();
        }

        return $dataTable->render('tasksassignmentmodule::index', $this->data);
    }

    /**
     * Show the form for creating a new resource.
     */
    public function create()
    {
        $this->pageTitle = __('app.menu.add_task_assignment');
        $this->view = 'tasksassignmentmodule::ajax.create';

        if (request()->ajax()) {
            return $this->returnAjax($this->view);
        }

        return view('tasksassignmentmodule::create', $this->data);
    }

    /**
     * Store a newly created resource in storage.
     */
    public function store(Request $request)
    {
        $validated = $request->validate([
            'date' => 'required|date',
            'task' => 'required|string|max:255',
            'assign_to' => 'required|string|max:100',
            'product' => 'required|string|max:100',
            'priority' => 'required|in:High,Medium,Low',
            'status' => 'required|in:Not Yet Started,In Progress,Completed',
            'eta' => 'nullable|string|max:50'
        ]);

        $task = new TaskAssignment();
        $task->fill($validated);
        $task->save();

        return Reply::successWithData(
            __('messages.recordSaved'),
            ['redirectUrl' => route('tasksassignmentmodule.index')]
        );
    }

    /**
     * Display the specified resource.
     */
    public function show($id)
    {
        $this->task = TaskAssignment::findOrFail($id);
        $this->pageTitle = __('app.task_assignment') . ' #' . $this->task->id;

        if (request()->ajax()) {
            return $this->returnAjax('tasksassignmentmodule::ajax.show');
        }

        return view('tasksassignmentmodule::show', $this->data);
    }

    /**
     * Show the form for editing the specified resource.
     */
    public function edit($id)
    {
        $this->task = TaskAssignment::findOrFail($id);
        $this->pageTitle = __('app.edit') . ' ' . __('app.task_assignment');

        if (request()->ajax()) {
            return $this->returnAjax('tasksassignmentmodule::ajax.edit');
        }

        return view('tasksassignmentmodule::edit', $this->data);
    }

    /**
     * Update the specified resource in storage.
     */
    public function update(Request $request, $id)
    {
        $task = TaskAssignment::findOrFail($id);

        $validated = $request->validate([
            'date' => 'required|date',
            'task' => 'required|string|max:255',
            'assign_to' => 'required|string|max:100',
            'product' => 'required|string|max:100',
            'priority' => 'required|in:High,Medium,Low',
            'status' => 'required|in:Not Yet Started,In Progress,Completed',
            'eta' => 'nullable|string|max:50'
        ]);

        $task->fill($validated);
        $task->save();

        return Reply::successWithData(
            __('messages.updateSuccess'),
            ['redirectUrl' => route('tasksassignmentmodule.index')]
        );
    }

    /**
     * Remove the specified resource from storage.
     */
    public function destroy($id)
    {
        TaskAssignment::destroy($id);
        return Reply::success(__('messages.deleteSuccess'));
    }

    /**
     * Mark task as started
     */
    public function start($id)
    {
        $task = TaskAssignment::findOrFail($id);
        $task->update([
            'status' => 'In Progress',
            'date' => now()
        ]);

        return Reply::success(__('Task marked as started'));
    }

    /**
     * Mark task as completed
     */
    public function complete($id)
    {
        $task = TaskAssignment::findOrFail($id);
        $task->update([
            'status' => 'Completed',
            'date' => now()
        ]);

        return Reply::success(__('Task marked as completed'));
    }
}