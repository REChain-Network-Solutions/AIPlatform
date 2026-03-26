<?php

namespace Modules\ComplianceIQ\Http\Controllers\Admin;

use Illuminate\Http\Request;
use Illuminate\Routing\Controller;
use Modules\ComplianceIQ\Entities\ComplianceHash;

class ComplianceLogController extends Controller
{
    public function index(Request $request)
    {
        $this->authorize('compliance.logs.view');
        $q = ComplianceHash::query()
            ->when($request->filled('status'), fn($x)=>$x->where('status',$request->status))
            ->when($request->filled('type'), fn($x)=>$x->where('hashable_type',$request->type))
            ->latest('computed_at');

        return view('complianceiq::admin.logs.index', [
            'logs' => $q->paginate(50),
        ]);
    }
}
