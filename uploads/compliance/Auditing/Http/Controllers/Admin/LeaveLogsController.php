<?php

namespace Modules\AuditLog\Http\Controllers\Admin;

use Illuminate\Http\Request;
use Maatwebsite\Excel\Facades\Excel;
use Modules\AuditLog\Exports\LeaveExport;
use Illuminate\Contracts\Support\Renderable;
use Modules\AuditLog\DataTables\LeaveLogDataTable;
use App\Http\Controllers\Admin\AdminBaseController;

class LeaveLogsController extends AdminBaseController
{
    /**
     * Display a listing of the resource.
     * @return Renderable
     */
    public function index(LeaveLogDataTable $dataTables)
    {
        $this->pageTitle = __('auditlog::app._log_activity.leaveLogs');
        dateRangeValidate();

        return $dataTables->render('auditlog::admin.leave.index', $this->data);
    }

    public function export()
    {
        dateRangeValidate();
        
        return Excel::download(new LeaveExport, 'leave-activities.xlsx');
    }

    /**
     * Show the form for creating a new resource.
     * @return Renderable
     */
    public function create()
    {
        return view('auditlog::create');
    }

    /**
     * Store a newly created resource in storage.
     * @param Request $request
     * @return Renderable
     */
    public function store(Request $request)
    {
        //
    }

    /**
     * Show the specified resource.
     * @param int $id
     * @return Renderable
     */
    public function show($id)
    {
        return view('auditlog::show');
    }

    /**
     * Show the form for editing the specified resource.
     * @param int $id
     * @return Renderable
     */
    public function edit($id)
    {
        return view('auditlog::edit');
    }

    /**
     * Update the specified resource in storage.
     * @param Request $request
     * @param int $id
     * @return Renderable
     */
    public function update(Request $request, $id)
    {
        //
    }

    /**
     * Remove the specified resource from storage.
     * @param int $id
     * @return Renderable
     */
    public function destroy($id)
    {
        //
    }
}
