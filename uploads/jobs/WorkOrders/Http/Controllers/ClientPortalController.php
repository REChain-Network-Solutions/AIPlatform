<?php
namespace Modules\WorkOrders\Http\Controllers;
use Illuminate\Routing\Controller; use Illuminate\Support\Facades\DB;
class ClientPortalController extends Controller{
  protected function clientId(){
    // Worksuite stores auth client on client guard; adapt as needed:
    if(auth('client')->check()){
      $user = auth('client')->user();
      // many builds expose $user->client_id; if not, fall back to id
      return $user->client_id ?? $user->id;
    }
    return null;
  }
  public function index(){
    $cid = $this->clientId(); abort_if(!$cid,403);
    $rows = DB::table('work_orders')->where('client_id',$cid)->orderByDesc('id')->limit(200)->get();
    return view('workorders::client.index', compact('rows'));
  }
  public function show($id){
    $cid = $this->clientId(); abort_if(!$cid,403);
    $wo = DB::table('work_orders')->where('id',$id)->where('client_id',$cid)->first(); abort_if(!$wo,404);
    $wc=DB::table('fsm_work_order_checks')->where('work_order_id',$id)->first();
    $items=$wc?DB::table('fsm_work_order_check_items')->where('work_check_id',$wc->id)->orderBy('id')->get():collect();
    $files=DB::table('work_order_client_files')->where('work_order_id',$id)->latest()->get();
    return view('workorders::client.show',compact('wo','wc','items','files'));
  }
}