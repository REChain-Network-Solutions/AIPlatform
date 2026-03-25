<?php

namespace Workdo\Pos\Entities;

use Carbon\Carbon;
use Carbon\CarbonPeriod;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Factories\HasFactory;
use App\Models\Permission;
use App\Models\Role;
use Rawilk\Settings\Support\Context;
use App\Models\User;
use App\Models\Purchase;
use App\Models\Warehouse;
use App\Models\WorkSpace;
use App\Models\Setting;
use App\Models\WarehouseProduct;

class Pos extends Model
{
    use HasFactory;

    protected $fillable = [
        'pos_id',
        'customer_id',
        'warehouse_id',
        'pos_date',
        'category_id',
        'status',
        'shipping_display',
        'created_by',
    ];

    public function customer()
    {
        if(module_is_active('Account'))
        {
            return $this->hasOne(\Workdo\Account\Entities\Customer::class, 'user_id', 'customer_id');
        }else{
            return $this->hasOne(User::class, 'id', 'customer_id');
        }
    }
    public function user(){
        return $this->hasOne(User::class, 'id', 'customer_id');
    }
    public function warehouse()
    {
        return $this->hasOne(Warehouse::class, 'id', 'warehouse_id');
    }

    public function posPayment(){
        return $this->hasOne(PosPayment::class,'pos_id','id');
    }

    public function PosProduct(){

        return $this->hasMany(PosProduct::class,'pos_id','pos_id');
    }

    public function items()
    {
        return $this->hasMany(PosProduct::class, 'pos_id', 'id');
    }
    public function itemswithproduct()
    {
        return $this->hasMany(PosProduct::class, 'pos_id', 'id')->with('product');
    }

    public function taxes()
    {
        return $this->hasOne('\Workdo\ProductService\Entities\Tax', 'id', 'tax');
    }

    public static function getallCategories()
    {
        $cat=[];
        if(module_is_active('ProductService'))
        {
            $cat = \Workdo\ProductService\Entities\Category::select('categories.*', \DB::raw("COUNT(pu.category_id) product_services"))
                ->leftjoin('product_services as pu','categories.id' ,'=','pu.category_id')
                ->where('categories.created_by', '=', creatorId())
                ->where('categories.workspace_id', '=', getActiveWorkSpace())
                ->where('categories.type', 0)
                ->orderBy('categories.id', 'DESC')->groupBy('categories.id')->get();
        }
        return $cat;

    }
    public static function getallproducts()
    {
        if(module_is_active('ProductService'))
        {
            return \Workdo\ProductService\Entities\ProductService::select('product_services.*', 'c.name as categoryname')
                ->leftjoin('categories as c', 'c.id', '=', 'product_services.category_id')
                ->where('product_services.created_by', '=', creatorId())
                ->where('product_services.workspace_id', '=', getActiveWorkSpace())
                ->orderBy('product_services.id', 'DESC');


        }
    }
    public static function posNumberFormat($number)
    {
        $data = !empty(company_setting('pos_prefix')) ? company_setting('pos_prefix') : '#SLO0000';
        return $data. sprintf("%05d", $number);
    }


    public static function customer_id($customer_name)
    {

        $customers = \DB::table('users')
            ->select(\DB::raw('IFNULL((SELECT id FROM users WHERE name = :name AND created_by = :created_by LIMIT 1), 0) AS customer_id'))
            ->addBinding(['name' => $customer_name, 'created_by' => creatorId()], 'select')
            ->first();

        return $customers->customer_id;
    }
    public static function tax_id($product_id)
    {
        if(module_is_active('ProductService'))
        {

            $results = \Workdo\ProductService\Entities\ProductService::where(['id'=>$product_id, 'created_by'=> creatorId(), 'workspace_id' => getActiveWorkSpace() ])->first();
            return $results->tax_id;
        }


    }

    public static function totalSelledAmount($month = false)
    {
        $sells = Pos::where('created_by', creatorId())
        ->where('workspace', getActiveWorkSpace());

        if ($month) {
            $sells = $sells->whereMonth('pos_date', '=', [date('m')]);
        }

        $sells = $sells->get()->groupBy('id', 'pos_date');

        $selledAmount = 0;
        if(!empty($sells) && count($sells) > 0)
        {
            foreach($sells as $day => $onesale)
            {
                foreach($onesale as $sale)
                {
                    $selledAmount += $sale->getTotal() + $sale->getTotalTax();
                }
            }
        }

        return currency_format_with_sym($selledAmount);

    }

    public function getTotalTax()
    {
        $totalTax = 0;
        foreach ($this->items as $product)
        {
            if(module_is_active('ProductService'))
            {
                $taxes = $this->totalTaxRate($product->tax);
            }
            else
            {
                $taxes = 0;
            }

            $totalTax += ($taxes / 100) * ($product->price * $product->quantity - $product->discount);
        }

        return $totalTax;
    }
    public function getTotal()
    {
        $subtotals = 0;
        foreach($this->items as $item)  //PosProduct
        {

            $subtotals += ($item->price * $item->quantity);

        }

        return $subtotals;
    }
    public static function getSalesReportChart()
    {
        $sales = Pos::whereDate('created_at', '>', Carbon::now()->subDays(10))->where('created_by', creatorId())->where('workspace',getActiveWorkSpace())->orderBy('created_at')->get()->groupBy(
                function ($val){
                    return Carbon::parse($val->created_at)->format('dm');
                }
            );
        $total = [];
        if(!empty($sales) && count($sales) > 0)
        {
            foreach($sales as $day => $onesale)
            {
                $totals = 0;
                foreach($onesale as $sale)
                {
                    $totals += $sale->getTotal() + $sale->getTotalTax();
                }
                $total[$day] = $totals;
            }
        }
        $m = date("m");
        $d = date("d");
        $y = date("Y");
        for($i = 0; $i <= 9; $i++)
        {
            $date                  = date('Y-m-d', mktime(0, 0, 0, $m, ($d - $i), $y));
            $salesArray['label'][] = $date;
            $date                  = date('dm', strtotime($date));
            $salesArray['value'][] = array_key_exists($date, $total) ? $total[$date] : 0;;
        }

        return $salesArray;
    }
    public static function templateData()
    {
        $arr              = [];
        $arr['colors']    = [
            '003580',
            '666666',
            '6676ef',
            'f50102',
            'f9b034',
            'fbdd03',
            'c1d82f',
            '37a4e4',
            '8a7966',
            '6a737b',
            '050f2c',
            '0e3666',
            '3baeff',
            '3368e6',
            'b84592',
            'f64f81',
            'f66c5f',
            'fac168',
            '46de98',
            '40c7d0',
            'be0028',
            '2f9f45',
            '371676',
            '52325d',
            '511378',
            '0f3866',
            '48c0b6',
            '297cc0',
            'ffffff',
            '000',
        ];
        $arr['templates'] = [
            "template1" => "New York",
            "template2" => "Toronto",
            "template3" => "Rio",
            "template4" => "London",
            "template5" => "Istanbul",
            "template6" => "Mumbai",
            "template7" => "Hong Kong",
            "template8" => "Tokyo",
            "template9" => "Sydney",
            "template10" => "Paris",
        ];
        return $arr;
    }

    public static function GivePermissionToRoles($role_id = null,$rolename = null)
    {
        $client_permissions=[
            'pos manage',
            'pos show',
            'pos add manage',
            'pos order manage',
        ];
        if($role_id != null)
        {
            if($rolename == 'client')
            {
                $roles_v = Role::where('name','client')->where('id',$role_id)->first();
                foreach($client_permissions as $permission_v){
                    $permission = Permission::where('name',$permission_v)->first();
                    if(!$roles_v->hasPermission($permission_v))
                    {
                        $roles_v->givePermission($permission);
                    }
                }
            }
        }
    }


    public static function tax($taxData)
    {

        $taxes  = [];
        if(!empty($taxData)){
            $taxArr = explode(',', $taxData);

            foreach($taxArr as $tax)
            {
                $taxes[] = \Workdo\ProductService\Entities\Tax::find($tax);
            }
        }


        return $taxes;
    }

    public static function taxRate($taxRate, $price, $quantity)
    {
        return ($taxRate / 100) * ($price * $quantity);
    }

    public static $taxes = null;
    public static function totalTaxRate($taxes)
    {
        if(is_null(self::$taxes)){
            self::$taxes  = \Workdo\ProductService\Entities\Tax::whereIn('id', explode(',', $taxes))->get();
        }

        return self::$taxes->sum('rate');
    }

    public function getSubTotal()
    {
        $subTotal = 0;
        foreach($this->items as $product)
        {

            $subTotal += ($product->price * $product->quantity);

        }

        return $subTotal;
    }

    public function getTotalDiscount()
    {
        $totalDiscount = 0;
        foreach($this->items as $product)
        {

            $totalDiscount += $product->discount;
        }

        return $totalDiscount;
    }

    public static function total_quantity($type, $quantity, $product_id)
    {
        if(module_is_active('ProductService'))
        {
            $product      = \Workdo\ProductService\Entities\ProductService::find($product_id);
            if(($product->type == 'product'))
            {
                $pro_quantity = $product->quantity;

                if($type == 'minus')
                {
                    $product->quantity = $pro_quantity - $quantity;
                }
                else
                {
                    $product->quantity = $pro_quantity + $quantity;
                }
                $product->save();
            }
        }
    }
    public static function addWarehouseStock($product_id, $quantity, $warehouse_id)
    {

        $product     = WarehouseProduct::where('product_id' , $product_id)->where('warehouse_id' , $warehouse_id)->first();

        if($product){
            $pro_quantity = $product->quantity;
            $product_quantity = $pro_quantity + $quantity;
        }else{
            $product_quantity = $quantity;
        }

        $data = WarehouseProduct::updateOrCreate(
            ['warehouse_id' => $warehouse_id, 'product_id' => $product_id,'created_by' => \Auth::user()->id,'workspace'=>getActiveWorkSpace()],
            ['warehouse_id' => $warehouse_id, 'product_id' => $product_id, 'quantity' => $product_quantity,'created_by' => \Auth::user()->id,'workspace'=>getActiveWorkSpace()])
          ;


    }

    public static function addProductStock($product_id, $quantity, $type, $description,$type_id)
    {
        $stocks                = new \Workdo\Account\Entities\StockReport();
        $stocks->product_id    = $product_id;
        $stocks->quantity	    = $quantity;
        $stocks->type          = $type;
        $stocks->type_id       = $type_id;
        $stocks->description   = $description;
        $stocks->workspace     = getActiveWorkSpace();
        $stocks->created_by    = creatorId();
        $stocks->save();
    }

     //quantity update in warehouse details
     public static function warehouse_quantity($type, $quantity, $product_id,$warehouse_id)
     {

        $product      = WarehouseProduct::where('warehouse_id',$warehouse_id)->where('product_id',$product_id)->first();
	if(isset($product->quantity) && !empty($product->quantity)){
		$pro_quantity = (!empty($product) && !empty($product->quantity))?$product->quantity:0;

		if($type == 'minus')
		{
		    $product->quantity = $pro_quantity!=0 ? $pro_quantity - $quantity : $quantity;
		}
		else
		{
		    $product->quantity = $pro_quantity + $quantity;

		}
		$product->save();
	}
     }

    //warehouse transfer
    public static function warehouse_transfer_qty($from_warehouse,$to_warehouse,$product_id,$quantity,$delete=null)
    {

        $toWarehouse      = WarehouseProduct::where('warehouse_id',$to_warehouse)->where('product_id',$product_id)->first();
        if(empty($toWarehouse)){
            if($delete != 'delete')
            {
                $transfer                = new WarehouseProduct();
                $transfer->warehouse_id  = $to_warehouse;
                $transfer->product_id    = $product_id;
                $transfer->quantity      = $quantity;
                $transfer->workspace     = getActiveWorkSpace();
                $transfer->created_by    = creatorId();
                $transfer->save();
            }
        }else{
            $toWarehouse->quantity   = $toWarehouse->quantity+$quantity;
            $toWarehouse->save();
        }
        $fromWarehouse               = WarehouseProduct::where('warehouse_id',$from_warehouse)->where('product_id',$product_id)->first();
        if(!empty($fromWarehouse))
        {
            $fromWarehouse->quantity     = ($fromWarehouse->quantity) - ($quantity);
            if($fromWarehouse->quantity <= 0){
                $fromWarehouse->delete();
            }
            else{
                $fromWarehouse->save();
            }
        }


    }

    public static function barcodeType()
    {
        $barcodeType = !empty(company_setting('barcode_type')) ? company_setting('barcode_type') : 'code128';
        return $barcodeType;
    }

    public static function barcodeFormat()
    {
        $barcodeFormat = !empty(company_setting('barcode_format')) ? company_setting('barcode_format') : 'css';
        return $barcodeFormat;
    }

}
