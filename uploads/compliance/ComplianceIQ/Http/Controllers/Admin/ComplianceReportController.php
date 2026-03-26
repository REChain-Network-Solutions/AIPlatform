<?php

namespace Modules\ComplianceIQ\Http\Controllers\Admin;

use Illuminate\Http\Request;
use Illuminate\Routing\Controller;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Response;
use Modules\ComplianceIQ\Entities\ComplianceReport;
use Modules\ComplianceIQ\Entities\ComplianceAnnotation;
use Modules\ComplianceIQ\Services\Export\ExporterFactory;

class ComplianceReportController extends Controller
{
    public function index(Request $request)
    {
        $this->authorize('compliance.view');
        $q = ComplianceReport::query()
            ->when($request->filled('status'), fn($x)=>$x->where('status',$request->status))
            ->latest('period_end');

        return view('complianceiq::admin.reports.index', [
            'reports' => $q->paginate(20),
        ]);
    }

    public function create()
    {
        $this->authorize('compliance.create');
        return view('complianceiq::admin.reports.create');
    }

    public function store(Request $request)
    {
        $this->authorize('compliance.create');
        $data = $request->validate([
            'title' => 'required|string|max:255',
            'period_start' => 'required|date',
            'period_end' => 'required|date|after_or_equal:period_start',
            'filters' => 'array',
        ]);
        $report = ComplianceReport::create($data + ['status'=>'draft']);
        return redirect()->route('admin.compliance.reports.show', $report)->with('status','Report created.');
    }

    public function show(ComplianceReport $report)
    {
        $this->authorize('compliance.view');
        return view('complianceiq::admin.reports.show', compact('report'));
    }

    public function edit(ComplianceReport $report)
    {
        $this->authorize('compliance.update');
        return view('complianceiq::admin.reports.edit', compact('report'));
    }

    public function update(Request $request, ComplianceReport $report)
    {
        $this->authorize('compliance.update');
        $data = $request->validate([
            'title' => 'required|string|max:255',
            'period_start' => 'required|date',
            'period_end' => 'required|date|after_or_equal:period_start',
            'filters' => 'array',
        ]);
        $report->update($data);
        return back()->with('status','Report updated.');
    }

    public function destroy(ComplianceReport $report)
    {
        $this->authorize('compliance.update');
        $report->delete();
        return redirect()->route('admin.compliance.reports.index')->with('status','Report deleted.');
    }

    public function export(ComplianceReport $report)
    {
        $this->authorize('compliance.export');
        [$name, $mime, $bin] = ExporterFactory::make(config('complianceiq.report_export.default'))->export($report);
        return response($bin, 200, [
            'Content-Type' => $mime,
            'Content-Disposition' => 'attachment; filename="'.$name.'"'
        ]);
    }, "report_{$report->id}.csv");
    }

    public function signoff(ComplianceReport $report)
    {
        $this->authorize('compliance.signoff');
        $report->update([
            'status' => 'signed_off',
            'signed_off_by' => Auth::id(),
            'signed_off_at' => now(),
        ]);
        return back()->with('status','Report signed off.');
    }

    public function annotate(Request $request, ComplianceReport $report)
    {
        $this->authorize('compliance.update');
        $data = $request->validate(['note'=>'required|string']);
        ComplianceAnnotation::create([
            'report_id' => $report->id,
            'user_id' => Auth::id(),
            'note' => $data['note'],
        ]);
        return back()->with('status','Annotation added.');
    }
}
