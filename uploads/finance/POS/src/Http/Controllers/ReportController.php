<?php

namespace Workdo\Pos\Http\Controllers;

use App\Models\User;
use Illuminate\Contracts\Support\Renderable;
use Illuminate\Http\Request;
use Illuminate\Routing\Controller;
use Workdo\Pos\Entities\Pos;
use App\Models\Purchase;
use App\Models\Warehouse;
use App\Models\WarehouseProduct;
use Carbon\Carbon;


class ReportController extends Controller
{

    public function yearMonth()
    {

        $month[] = __('January');
        $month[] = __('February');
        $month[] = __('March');
        $month[] = __('April');
        $month[] = __('May');
        $month[] = __('June');
        $month[] = __('July');
        $month[] = __('August');
        $month[] = __('September');
        $month[] = __('October');
        $month[] = __('November');
        $month[] = __('December');

        return $month;
    }
    public function yearList()
    {
        $starting_year = date('Y', strtotime('-5 year'));
        $ending_year   = date('Y');

        foreach(range($ending_year, $starting_year) as $year)
        {
            $years[$year] = $year;
        }

        return $years;
    }



    public function posDailyReport(Request $request)
    {

        if(\Auth::user()->isAbleTo('report pos'))
        {
            $warehouse     = Warehouse::where('created_by', '=', creatorId())->where('workspace',getActiveWorkSpace())->get()->pluck('name', 'id');
            $warehouse->prepend('All Warehouse',0);

            $query = Pos::where('created_by', '=', creatorId())->where('workspace',getActiveWorkSpace());

            $customer     =  User::where('type','client')->where('created_by', creatorId())->where('workspace_id',getActiveWorkSpace())->get()->pluck('name', 'id');
            $customer->prepend('All Customer', 0);

            if(!empty($request->warehouse))
            {
                $query->where('warehouse_id', '=', $request->warehouse);
            }
            if(!empty($request->customer))
            {
                $query->where('customer_id', '=', $request->customer);
            }

            $arrDuration = [];
            $data=[];
            if(!empty($request->start_date) && !empty($request->end_date))
            {
                $first_date=$request->start_date;
                $end_date=$request->end_date;
            }
            else
            {
                $first_date=date('Y-m-d', strtotime('today - 30 days'));
                $end_date=date('Y-m-d', strtotime('today - 0 days'));

            }
            $query->whereBetween('pos_date', [$first_date, $end_date]);
            $poses = $query->get()->groupBy(
                function ($val) {
                    return Carbon::parse($val->pos_date)->format('Y-m-d');
                });
            $total = [];
            if (!empty($poses) && count($poses) > 0) {
                foreach ($poses as $day => $onepos) {
                    $totals = 0;
                    foreach ($onepos as $pos) {

                        $totals += $pos->getTotal() + $pos->getTotalTax();
                    }
                    $total[$day] = $totals;
                }
            }
            if(!empty($request->start_date) && !empty($request->end_date)){
                $previous_days = strtotime($request->start_date . " -1 day");
                for($i = 0; $i < 30; $i++)
                {
                    $previous_days = strtotime(date('Y-m-d', $previous_days) . " +1 day");
                    $arrDuration[] = date('d-M', $previous_days);
                    $date=date('Y-m-d', $previous_days);
                    $data[]=isset($total[$date])?$total[$date]:0;
                }
            }else{
                $previous_days = strtotime("-1 month +1 days");
                for($i = 0; $i < 30; $i++)
                {
                    $previous_days = strtotime(date('Y-m-d', $previous_days) . " +1 day");
                    $arrDuration[] = date('d-M', $previous_days);
                    $date=date('Y-m-d', $previous_days);
                    $data[]=isset($total[$date])?$total[$date]:0;
                }
            }


            $filter['startDate'] =  $first_date;
            $filter['endDate']   =  $end_date;
            $warehouses = warehouse::where('id', '=', $request->warehouse)->where('created_by', creatorId())->first();
            $filter['warehouse']       = !empty($warehouses)?$warehouses->name:'';
            $customers     =  User::where('type','client')->where('id', '=', $request->customer)->where('created_by', creatorId())->where('workspace_id',getActiveWorkSpace())->first();
            $filter['customer']       = !empty($customers)?$customers->name:'';

            return view('pos::report.daily_pos',compact('warehouse','customer','arrDuration','data','filter'));
        }
        else
        {
            return redirect()->back()->with('error', __('Permission denied.'));
        }

    }

    public function posMonthlyReport(Request $request)
    {
        if(\Auth::user()->isAbleTo('report pos'))
        {
            $monthList = $this->yearMonth();
            $yearList = $this->yearList();

            $warehouse     = warehouse::where('created_by', creatorId())->get()->pluck('name', 'id');
            $warehouse->prepend('All Warehouse',0);


            $customer     =  User::where('type','client')->where('created_by', creatorId())->where('workspace_id',getActiveWorkSpace())->get()->pluck('name', 'id');
            $customer->prepend('All Customer', 0);

            $query = Pos::where('created_by', '=', creatorId());
            if(!empty($request->warehouse))
            {
                $query->where('warehouse_id', '=', $request->warehouse);
            }
            if(!empty($request->customer))
            {
                $query->where('customer_id', '=', $request->customer);
            }
            $arrDuration = [];
            $data=[];
            if(!empty($request->year))
            {
                $year= $request->year;
            }
            else
            {
                $year=date('Y');
            }
            $query->whereYear('pos_date', $year);
            $poses = $query->get()->groupBy(
                function ($val) {
                    return Carbon::parse($val->pos_date)->format('m');
                });
            $total = [];
            if (!empty($poses) && count($poses) > 0) {
                foreach ($poses as $month => $onepos) {
                    $totals = 0;
                    foreach ($onepos as $pos) {
                        $totals += $pos->getTotal() + $pos->getTotalTax();
                    }
                    $total[$month] = $totals;
                }
            }
            for($i = 1; $i <= 12; $i++)
            {

                $dateFormat=date('Y-'.$i.'-01');
                $arrDuration[] = date("my", strtotime($dateFormat));
                $month1=date("m", strtotime($dateFormat));
                $data[]=isset($total[$month1])?$total[$month1]:0;

            }



            $filter['startMonth'] = 'Jan-' . $year;
            $filter['endMonth']   = 'Dec-' . $year;
            $warehouses = warehouse::where('id', '=', $request->warehouse)->where('created_by', creatorId())->first();
            $filter['warehouse']       = !empty($warehouses)?$warehouses->name:'';
            $customers     =  User::where('type','client')->where('id', '=', $request->customer)->where('created_by', creatorId())->where('workspace_id',getActiveWorkSpace())->first();
            $filter['customer']       = !empty($customers)?$customers->name:'';

            return view('pos::report.monthly_pos',compact('monthList','yearList','warehouse','customer','arrDuration','data','filter'));

        }
        else
        {
            return redirect()->back()->with('error', __('Permission denied.'));
        }
    }

    public function posVsPurchaseReport(Request $request)
    {
        if(\Auth::user()->isAbleTo('report pos'))
        {
            $data['monthList'] = $month = $this->yearMonth();
            $data['yearList']  = $this->yearList();

            if(isset($request->year))
            {
                $year = $request->year;
            }
            else
            {
                $year = date('Y');
            }
            $data['currentYear'] = $year;


            // ------------------------------TOTAL POS-----------------------------------------------------------

            $posData = Pos:: selectRaw('MONTH(pos_date) as month,YEAR(pos_date) as year,pos_id,id')
                ->where('created_by', creatorId());
            $posData->whereRAW('YEAR(pos_date) =?', [$year]);

            $posData        = $posData->get();
            $posTotalArray = [];
            foreach($posData as $pos)
            {
                $posTotalArray[$pos->month] = 0;
            }
            foreach($posData as $pos)
            {
                $posTotalArray[$pos->month] += $pos->getTotal() + $pos->getTotalTax();
            }


            // ------------------------------ TOTAL PAYMENT-----------------------------------------------------------
            $purchaseData = Purchase:: selectRaw('MONTH(send_date) as month,YEAR(send_date) as year,purchase_id,id')
                ->where('created_by', creatorId())
                ->where('status', '!=', 0);
            $purchaseData->whereRAW('YEAR(send_date) =?', [$year]);
            $purchaseData        = $purchaseData->get();
            $purchaseTotalArray = [];
            foreach($purchaseData as $purchase)
            {
                $purchaseTotalArray[$purchase->month] = 0;
            }
            foreach($purchaseData as $purchase)
            {
                $purchaseTotalArray[$purchase->month] += $purchase->getTotal();
            }

            //            -----------------------------

            for($i = 1; $i <= 12; $i++)
            {
                $PosTotal[] = array_key_exists($i, $posTotalArray) ? $posTotalArray[$i] : 0;
                $PurchaseTotal[] = array_key_exists($i, $purchaseTotalArray) ? $purchaseTotalArray[$i] : 0;

            }
            $totalPos = array_map(
                function (){
                    return array_sum(func_get_args());
                }, $PosTotal
            );

            $totalPurchase = array_map(
                function (){
                    return array_sum(func_get_args());
                }, $PurchaseTotal
            );


            $profits = [];
            $keys   = array_keys($totalPos + $totalPurchase);
            foreach($keys as $v)
            {
                $profits[$v] = number_format((empty($totalPos[$v]) ? 0 : $totalPos[$v]) - (empty($totalPurchase[$v]) ? 0 : $totalPurchase[$v]), 2);
            }

            $data['posTotal']        = $PosTotal;
            $data['purchaseTotal']        = $PurchaseTotal;
            $data['profits']              = $profits;
            $filter['startDateRange'] = 'Jan-' . $year;
            $filter['endDateRange']   = 'Dec-' . $year;


            return view('pos::report.pos_vs_purchase', compact('filter'),  $data);
        }
        else
        {
            return redirect()->back()->with('error', __('Permission denied.'));
        }
    }

}
