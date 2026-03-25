<?php
namespace Modules\WorkOrders\Http\Controllers; use Illuminate\Routing\Controller; use Illuminate\Http\Request; use Illuminate\Support\Facades\DB; use Illuminate\Support\Facades\Storage;
class ClientSignController extends Controller{protected function clientId(){return auth('client')->user()->client_id ?? auth('client')->id();}
  public function form($id){ $wo=DB::table('work_orders')->where('id',$id)->where('client_id',$this->clientId())->first(); abort_if(!$wo,404); return view('workorders::client/sign', compact('wo')); }
  public function submit(Request $r,$id){
    $data=$r->validate(['name'=>'required|string|max:120','signature_data'=>'required|string']);
    $png=base64_decode(preg_replace('/^data:image\/(png|jpeg);base64,/','',$data['signature_data']));
    $file='signatures/wo_'.$id.'_'.time().'.png';
    Storage::disk('public')->put($file,$png);
    DB::table('work_orders')->where('id',$id)->update(['client_signed_at'=>now(),'client_signature_path'=>'storage/'.$file,'client_sign_name'=>$data['name'],'updated_at'=>now()]);
    return redirect()->route('client.workorders.show',$id)->with('status','Thanks, your sign-off has been recorded.');
  }
}