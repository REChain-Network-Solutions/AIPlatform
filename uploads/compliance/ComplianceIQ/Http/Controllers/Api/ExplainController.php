<?php

namespace Modules\ComplianceIQ\Http\Controllers\Api;

use Illuminate\Http\Request;
use Illuminate\Routing\Controller;
use Modules\ComplianceIQ\Services\AI\ComplianceAIInterface;

class ExplainController extends Controller
{
    public function __construct(protected ComplianceAIInterface $ai) {}
    public function explain(Request $request)
    {
        $request->validate([
            'issue' => 'required|string|min:10',
            'context' => 'array'
        ]);

        $result = $this->ai->explainIssue($request->issue, $request->context ?? []);
        return response()->json($result);
    }
}
