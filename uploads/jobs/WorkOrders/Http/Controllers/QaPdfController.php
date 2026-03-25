<?php
namespace Modules\WorkOrders\Http\Controllers; use Illuminate\Routing\Controller; use Illuminate\Support\Facades\DB; use Barryvdh\DomPDF\Facade\Pdf; use Modules\WorkOrders\Services\BrandingService;
class QaPdfController extends Controller{
  public function show($id){
    $wo=DB::table('work_orders')->where('id',$id)->first(); abort_if(!$wo,404);
    $wc=DB::table('fsm_work_order_checks')->where('work_order_id',$id)->first();
    $items=$wc?DB::table('fsm_work_order_check_items')->where('work_check_id',$wc->id)->orderBy('id')->get():collect();
    $brand=BrandingService::get(); $html=view('workorders::pdf.qa', compact('wo','wc','items','brand'))->render();
    $pdf=Pdf::loadHTML($html); return $pdf->stream("WO_{$id}_QA.pdf");
  }
}