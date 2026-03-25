<?php

namespace Modules\BudgetAllocationAprovalModule\Http\Controllers;

use App\Helper\Reply;
use App\Http\Controllers\AccountBaseController;
use App\Models\BudgetAllocationApproval;
use Illuminate\Http\Request;
use App\DataTables\BudgetAllocationDataTable;

class BudgetAllocationModuleController extends AccountBaseController
{
    public function __construct()
    {
        parent::__construct();
        $this->pageTitle = 'app.menu.budget_allocation';

        $this->middleware(function ($request, $next) {
            abort_if(!in_array('clients', $this->user->modules), 403);
            return $next($request);
        });
    }

    /**
     * Display a listing of the resource.
     */
    public function index(BudgetAllocationDataTable $dataTable)
    {
        if (!request()->ajax()) {
            $this->budget_allocations = BudgetAllocationApproval::all();
        }

        return $dataTable->render('budgetallocationaprovalmodule::index', $this->data);
    }

    /**
     * Show the form for creating a new resource.
     */
    public function create()
    {
        $this->pageTitle = __('app.menu.add_budget_allocation');
        $this->view = 'budgetallocationaprovalmodule::ajax.create';

        if (request()->ajax()) {
            return $this->returnAjax($this->view);
        }

        return view('budgetallocationaprovalmodule::create', $this->data);
    }

    /**
     * Store a newly created resource in storage.
     */
    public function store(Request $request)
    {
        $validated = $request->validate([
            'project_id' => 'required|string|max:50',
            'project_name' => 'required|string|max:255',
            'department' => 'required|string|max:255',
            'budget_requested' => 'required|numeric|min:0',
            'approval_status' => 'required|in:pending,approved,rejected',
            'budget_approved' => 'nullable|numeric|min:0',
            'approval_date' => 'nullable|date',
            'comments' => 'nullable|string'
        ]);

        $allocation = new BudgetAllocationApproval();
        $allocation->fill($validated);

        // Only set approved amount and date if status is approved
        if ($request->approval_status === 'approved') {
            $allocation->budget_approved = $request->budget_approved ?? $request->budget_requested;
            $allocation->approval_date = $request->approval_date ?? now();
        }

        $allocation->save();

        return Reply::successWithData(
            __('messages.recordSaved'),
            ['redirectUrl' => route('budgetallocationaprovalmodule.index')]
        );
    }

    /**
     * Display the specified resource.
     */
    public function show($id)
    {
        $this->allocation = BudgetAllocationApproval::findOrFail($id);
        $this->pageTitle = __('app.budget_allocation') . ' #' . $this->allocation->id;
        $this->view = 'budgetallocationaprovalmodule::ajax.show';
        if (request()->ajax()) {
            return $this->returnAjax($this->view);
        }

        return view('budgetallocationaprovalmodule::show', $this->data);
    }

    /**
     * Show the form for editing the specified resource.
     */
    public function edit($id)
    {
        $this->allocation = BudgetAllocationApproval::findOrFail($id);
        $this->pageTitle = __('app.edit') . ' ' . __('app.budget_allocation');

        if (request()->ajax()) {
            return $this->returnAjax('budgetallocationaprovalmodule::ajax.edit');
        }

        return view('budgetallocationaprovalmodule::edit', $this->data);
    }

    /**
     * Update the specified resource in storage.
     */
    public function update(Request $request, $id)
    {
        $allocation = BudgetAllocationApproval::findOrFail($id);

        $validated = $request->validate([
            'project_name' => 'required|string|max:255',
            'department' => 'required|string|max:255',
            'budget_requested' => 'required|numeric|min:0',
            'approval_status' => 'required|in:pending,approved,rejected',
            'budget_approved' => 'nullable|numeric|min:0',
            'comments' => 'nullable|string'
        ]);

        $allocation->fill($validated);

        // Update approval details if status changed to approved
        if ($request->approval_status === 'approved' && $allocation->isDirty('approval_status')) {
            $allocation->approval_date = now();
            $allocation->approved_by = $this->user->id;
        }

        $allocation->save();

        return Reply::successWithData(
            __('messages.updateSuccess'),
            ['redirectUrl' => route('budgetallocationaprovalmodule.index')]
        );
    }

    /**
     * Remove the specified resource from storage.
     */
    public function destroy($id)
    {
        BudgetAllocationApproval::destroy($id);
        return Reply::success(__('messages.deleteSuccess'));
    }

    /**
     * Approve the specified budget allocation.
     */
    public function approve($id)
    {
        $allocation = BudgetAllocationApproval::findOrFail($id);
        $allocation->update([
            'approval_status' => 'approved',
            'budget_approved' => $allocation->budget_requested,
            'approval_date' => now(),
            'approved_by' => $this->user->id
        ]);

        return Reply::success(__('Budget approved successfully'));
    }

    /**
     * Reject the specified budget allocation.
     */
    public function reject($id)
    {
        BudgetAllocationApproval::findOrFail($id)->update([
            'approval_status' => 'rejected',
            'approval_date' => now(),
            'approved_by' => $this->user->id
        ]);

        return Reply::success(__('Budget rejected'));
    }
}