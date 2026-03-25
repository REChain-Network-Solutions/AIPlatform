<?php
namespace Modules\WorkOrders\Http\Middleware;
use Closure; use Illuminate\Support\Facades\DB;

class EnsureCompanyScope {
  public function handle($request, Closure $next, $table='work_orders', $param='id'){
    $id = $request->route($param);
    if(!$id) return $next($request);
    $row = DB::table($table)->where('id',$id)->first();
    if(!$row) abort(404);
    $user = auth()->user();
    $companyId = method_exists($user,'company_id') ? $user->company_id : ($user->company->id ?? null);
    if(!$companyId || (property_exists($row,'company_id') && $row->company_id != $companyId)){
      abort(403, 'Cross-tenant access denied.');
    }
    return $next($request);
  }
}
