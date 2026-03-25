<?php
namespace Modules\WorkOrders\Policies;
use Illuminate\Auth\Access\HandlesAuthorization; use Illuminate\Support\Facades\DB;

class WorkOrderPolicy {
  use HandlesAuthorization;

  protected function sameCompany($user, $woId): bool {
    $wo = DB::table('work_orders')->where('id',$woId)->first();
    if(!$wo) return false;
    $companyId = method_exists($user,'company_id') ? $user->company_id : ($user->company->id ?? null);
    return $companyId && $wo->company_id == $companyId;
  }

  public function view($user, $woId): bool {
    return $this->sameCompany($user,$woId);
  }

  public function start($user, $woId): bool {
    if(!$this->sameCompany($user,$woId)) return false;
    // Enforce compliance chain
    $okDocs = DB::table('fsm_compliance_docs as c')
      ->join('work_orders as w','w.company_id','=','c.company_id')
      ->where('w.id',$woId)->where('c.expires_at','>',now())->exists();
    $permitApproved = DB::table('fsm_work_permits')->where('wo_id',$woId)->where('status','approved')->exists();
    $inspectionPass = DB::table('fsm_inspections')->where('wo_id',$woId)->where('passed',1)->exists();
    // If tables not present yet, treat missing as true to avoid blocking dev
    $softOk = function($exists, $defaultTrue){ return $exists === null ? $defaultTrue : $exists; };
    return ($okDocs || !DB::getSchemaBuilder()->hasTable('fsm_compliance_docs'))
        && ($permitApproved || !DB::getSchemaBuilder()->hasTable('fsm_work_permits'))
        && ($inspectionPass || !DB::getSchemaBuilder()->hasTable('fsm_inspections'));
  }

  public function complete($user, $woId): bool {
    return $this->sameCompany($user,$woId);
  }

  public function invoice($user, $woId): bool {
    return $this->sameCompany($user,$woId);
  }

  public function approve($user, $woId): bool {
    return $this->sameCompany($user,$woId);
  }
}
