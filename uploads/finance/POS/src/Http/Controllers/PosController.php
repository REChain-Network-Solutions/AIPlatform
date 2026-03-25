<?php

namespace Workdo\Pos\Http\Controllers;

use Illuminate\Contracts\Support\Renderable;
use Illuminate\Http\Request;
use Illuminate\Routing\Controller;
use Illuminate\Support\Facades\Validator;
use Rawilk\Settings\Support\Context;
use App\Models\Purchase;
use App\Models\Warehouse;
use Workdo\Pos\Entities\Pos;
use App\Models\WarehouseProduct;
use Workdo\Pos\Entities\PosProduct;
use Workdo\Pos\Entities\PosPayment;
use Workdo\Pos\Entities\PosUtility;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Crypt;
use App\Models\User;
use App\Models\Setting;
use App\Models\WorkSpace;
use Workdo\Pos\DataTables\PosOrderDataTable;
use DB;
use Workdo\Pos\Events\CreatePaymentPos;
use Stripe\Product;
use Workdo\Pos\DataTables\BarcodeDataTable;
use Workdo\Quotation\Entities\Quotation;
use Workdo\Quotation\Entities\QuotationProduct;
use Workdo\Account\Entities\Customer;
class PosController extends Controller
{
    /**
     * Display a listing of the resource.
     * @return Renderable
     */
    public function __construct()
    {
        if(module_is_active('GoogleAuthentication'))
        {
            $this->middleware('2fa');
        }
    }
    public function index(Request $request)
    {
        if (\Auth::user()->isAbleTo('pos add manage'))
        {
            session()->forget('pos');
            $customers=[];
            $customers      = User::where('type','client')->where('created_by', creatorId())->where('workspace_id',getActiveWorkSpace())->get()->pluck('name', 'name');
            $customers->prepend('Walk-in-customer', 'Walk-in-customer');
            $user = \Auth::user();

            $details = [
                'pos_id' =>Pos::posNumberFormat($this->invoicePosNumber()),
                'customer' => $customers != null ? $customers->toArray() : [],
                'user' => $user != null ? $user->toArray() : [],
                'date' => date('Y-m-d'),
                'pay' => 'show',
            ];
            $warehouses = warehouse::select('*', \DB::raw("CONCAT(name) AS name"))->where('created_by', creatorId())->where('workspace',getActiveWorkSpace())->get()->pluck('name', 'id');

            $cart = session()->get('pos');
            if (isset($cart) && count($cart) > 0)
            {
                if(module_is_active('ProductService'))
                {
                    $product = \Workdo\ProductService\Entities\ProductService::where('created_by', creatorId())->where('workspace_id',getActiveWorkSpace())->whereIn('id',array_keys($cart))->get();
                    if(count($product) == 0)
                    {
                        session()->forget('pos');
                    }
                }
            }

            if (isset($request->quotation_id)) {
                $quotation = Quotation::find($request->quotation_id);
                $customer = '';
                $warehouseId = '';
                if($quotation)
                {
                    $customer= Customer::where('user_id',$quotation->customer_id)->first();
                    if(empty($customer))
                    {
                        $customer = User::find($quotation->customer_id);
                    }
                    $customer = $customer->name;

                    $warehouseId = $quotation->warehouse_id;

                    $quotationProduct = QuotationProduct::where('quotation_id', $request->quotation_id)->get();

                    foreach ($quotationProduct as $value) {
                        $products = Quotation::quotationProduct($value);
                    }
                }
            } else {
                $customer = '';
                $warehouseId = '';
            }

            $id = !empty($request->quotation_id) ? $request->quotation_id : '0';

            return view('pos::pos.index',compact('customers','warehouses','details','customer', 'warehouseId', 'id'));        
        }
        else
        {
            return redirect()->back()->with('error', __('Permission denied.'));
        }
    }

    /**
     * Show the form for creating a new resource.
     * @return Renderable
     */
    public function create(Request $request)
    {
        $sess = session()->get('pos');
            if (Auth::user()->isAbleTo('pos add manage') && isset($sess) && !empty($sess) && count($sess) > 0) {
            $user = Auth::user();

            if(module_is_active('Account'))
            {
                $user =  User::where('name', '=', $request->vc_name)->where('type','client')->where('created_by', creatorId())->where('workspace_id',getActiveWorkSpace())->first();
                if(!empty($user->id))
                {

                    $customer = \Workdo\Account\Entities\Customer::where('user_id',$user->id)->where('name', '=', $request->vc_name)->where('created_by', creatorId())->where('workspace',getActiveWorkSpace())->first();
                    if(!empty($customer)){
                        $customer = $customer;
                    }
                    else{
                        $customer = $user;
                    }
                }
                else{
                    $customer = NULL;
                    $user = Auth::user();
                }
            }
            else
            {
                $customer      = User::where('name', '=', $request->vc_name)->where('type','client')->where('created_by', creatorId())->where('workspace_id',getActiveWorkSpace())->first();
            }
	    $user = \Auth::user();
            $warehouse = warehouse::where('id', '=', $request->warehouse_name)->where('created_by', creatorId())->where('created_by', creatorId())->where('workspace',getActiveWorkSpace())->first();

            $details = [
                'pos_id' => Pos::posNumberFormat($this->invoicePosNumber()),
                'customer' => $customer != null ? $customer->toArray() : [],
                'warehouse' => $warehouse != null ? $warehouse->toArray() : [],
                'user' => $user != null ? $user->toArray() : [],
                'date' => date('Y-m-d'),
                'pay' => 'show',
            ];
            if (!empty($details['customer']['billing_state']))
            {

                $warehousedetails = '<h7 class="text-dark">' . ucfirst($details['warehouse']['name'])  . '</p></h7>';
                $details['customer']['billing_state'] = $details['customer']['billing_state'] != '' ? ", " . $details['customer']['billing_state'] : '';
                $details['customer']['shipping_state'] = $details['customer']['shipping_state'] != '' ? ", " . $details['customer']['shipping_state'] : '';

                $customerdetails = '<h6 class="text-dark">' . ucfirst($details['customer']['name']) . '<p class="m-0 h6 font-weight-normal">' . $details['customer']['billing_phone'] . '</p>' . '<p class="m-0 h6 font-weight-normal">' . $details['customer']['billing_address'] . '</p>' . '<p class="m-0 h6 font-weight-normal">' . $details['customer']['billing_city'] . $details['customer']['billing_state'] . '</p>' . '<p class="m-0 h6 font-weight-normal">' . $details['customer']['billing_country'] . '</p>' . '<p class="m-0 h6 font-weight-normal">' . $details['customer']['billing_zip'] . '</p></h6>';

                $shippdetails = '<h6 class="text-dark"><b>' . ucfirst($details['customer']['name']) . '</b>' . '<p class="m-0 h6 font-weight-normal">' . $details['customer']['shipping_phone'] . '</p>' . '<p class="m-0 h6 font-weight-normal">' . $details['customer']['shipping_address'] . '</p>' . '<p class="m-0 h6 font-weight-normal">' . $details['customer']['shipping_city'] . $details['customer']['shipping_state'] . '</p>' . '<p class="m-0 h6 font-weight-normal">' . $details['customer']['shipping_country'] . '</p>' . '<p class="m-0 h6 font-weight-normal">' . $details['customer']['shipping_zip'] . '</p></h6>';

            }
            else {

                if (!empty($details['customer']))
                {
                    $customerdetails = '<h6 class="text-dark">' . ucfirst($details['customer']['name']) .  '</p></h6>';
                }
                else{

                    $customerdetails = '<h2 class="h6"><b>' . __('Walk-in Customer') . '</b><h2>';
                }
                $warehousedetails = '<h7 class="text-dark">' . ucfirst($details['warehouse']['name'])  . '</p></h7>';
                $shippdetails = '-';

            }

            $settings['company_telephone'] = company_setting('company_telephone') != '' ? ", " . company_setting('company_telephone') : '';
            $settings['company_state']     = company_setting('company_state') != '' ? ", " . company_setting('company_state') : '';

            $userdetails = '<h6 class="text-dark"><b>' . ucfirst($details['user']['name']) . ' </b> <h2  class="font-weight-normal">' . '<p class="m-0 font-weight-normal">' . company_setting('company_name') . '</p>' . '<p class="m-0 font-weight-normal">' . company_setting('company_telephone') . '</p>' . '<p class="m-0 font-weight-normal">' . company_setting('company_address') . '</p>' . '<p class="m-0 h6 font-weight-normal">' . company_setting('company_city') . ', ' . company_setting('company_state') . '</p>' . '<p class="m-0 font-weight-normal">' . company_setting('company_country') . '</p>' . '<p class="m-0 font-weight-normal">' . company_setting('company_zipcode') . '</p></h2>';

            $details['customer']['details'] = $customerdetails;
            $details['warehouse']['details'] = $warehousedetails;
            $details['customer']['shippdetails'] = $shippdetails;

            $details['user']['details'] = $userdetails;

            $mainsubtotal = 0;
            $sales        = [];
            foreach ($sess as $key => $value) {
                $subtotal = $value['price'] * $value['quantity'];
                $tax      = ($subtotal * $value['tax']) / 100;
                $sales['data'][$key]['name']       = $value['name'];
                $sales['data'][$key]['quantity']   = $value['quantity'];
                $sales['data'][$key]['price']      = currency_format_with_sym($value['price']);
                $sales['data'][$key]['tax']        = $value['tax'] . '%';
                $sales['data'][$key]['product_tax'] = $value['product_tax'];
                $sales['data'][$key]['tax_amount'] = currency_format_with_sym($tax);
                $sales['data'][$key]['subtotal']   = currency_format_with_sym($value['subtotal']);
                $mainsubtotal                      += $value['subtotal'];
            }

            if($request->discount <= $mainsubtotal){
                $discount=!empty($request->discount)?$request->discount:0;
            }
            else{
                $discount=$mainsubtotal;
            }

            $sales['discount'] = currency_format_with_sym($discount);
            $total= $mainsubtotal-$discount;
            $sales['sub_total'] = currency_format_with_sym($mainsubtotal);
            $sales['total'] = currency_format_with_sym($total);
            return view('pos::pos.show', compact('sales', 'details'));
        } else {
            return response()->json(
                [
                    'error' => __('Add some products to cart!'),
                ],
                '404'
            );
        }
    }

    /**
     * Store a newly created resource in storage.
     * @param Request $request
     * @return Renderable
     */
    public function store(Request $request)
    {
        $discount=(!empty($request->discount) && $request->discount > 0 ) ? $request->discount : 0;
        if (Auth::user()->isAbleTo('pos add manage')) {

            if ($request->quotation_id != 0) {
                $quotation = Quotation::where('id', $request->quotation_id)->first();
                $quotation->is_converted = 1;
                $quotation->save();
            }
            $user_id = creatorId();
            $customer_id      = Pos::customer_id($request->vc_name);
            $warehouse_id      = warehouse::warehouse_id($request->warehouse_name);
            $pos_id       = $this->invoicePosNumber();
            $sales            = session()->get('pos');
            if (isset($sales) && !empty($sales) && count($sales) > 0) {
                $result = DB::table('pos')->where('pos_id', $pos_id)->where('created_by', $user_id)->where('workspace',getActiveWorkSpace())->get();
                if (count($result) > 0) {
                    return response()->json(
                        [
                            'code' => 200,
                            'success' => __('Payment is already completed!'),
                        ]
                    );
                } else {
                    $pos = new Pos();
                    $pos->pos_id           = $pos_id;
                    $pos->customer_id      = $customer_id;
                    $pos->warehouse_id     = $request->warehouse_name;
                    $pos->workspace        = getActiveWorkSpace();
                    $pos->pos_date         = date('Y-m-d');
                    $pos->created_by       = $user_id;
                    $pos->save();

                    if ($request->quotation_id != 0) {
                        $quotation->converted_pos_id = $pos->id;
                        $quotation->save();
                    }
                    foreach ($sales as $key => $value) {
                        $product_id = $value['id'];
                        $product = \Workdo\ProductService\Entities\ProductService::whereId($product_id)->where('created_by', $user_id)->where('workspace_id',getActiveWorkSpace())->first();
                        $original_quantity = ($product == null) ? 0 : (int)$product->quantity;
                        $product_quantity = $original_quantity - $value['quantity'];
                        if ($product != null && !empty($product)) {
                            \Workdo\ProductService\Entities\ProductService::where('id', $product_id)->update(['quantity' => $product_quantity]);
                        }
                        $tax_id =Pos::tax_id($product_id);
                            $positems = new PosProduct();
                            $positems->pos_id     = $pos->id;
                            $positems->product_id = $product_id;
                            $positems->price      = $value['price'];
                            $positems->quantity   = $value['quantity'];
                            $positems->tax        = $tax_id;
                            $positems->discount        = $discount;
                            $positems->workspace =getActiveWorkSpace();
                            $positems->save();
                            Pos::warehouse_quantity('minus',$positems->quantity,$positems->product_id,$request->warehouse_name);
                        if(module_is_active('Account'))
                        {
                            $type='Pos';
                            $type_id = $pos->id;
                            \Workdo\Account\Entities\StockReport::where('type','=','pos')->where('type_id' ,'=', $pos->id)->delete();
                            $description=$positems->quantity.'  '.__(' quantity sold in pos').' '. Pos::posNumberFormat($pos->pos_id);
                            Pos::addProductStock( $positems->product_id,$positems->quantity,$type,$description,$type_id);
                        }
                    }
                    $posPayment                 = new PosPayment();
                    $posPayment->pos_id          =$pos->id;
                    $posPayment->date           = $request->date;
                    $mainsubtotal = 0;
                    $sales        = [];
                    $sess = session()->get('pos');
                    if(isset($sess) && !empty($sess) && count($sess) > 0){
                        foreach ($sess as $key => $value) {
                            $subtotal = $value['price'] * $value['quantity'];
                            $tax      = ($subtotal * $value['tax']) / 100;
                            $sales['data'][$key]['price']      = currency_format_with_sym($value['price']);
                            $sales['data'][$key]['tax']        = $value['tax'] . '%';
                            $sales['data'][$key]['tax_amount'] = currency_format_with_sym($tax);
                            $sales['data'][$key]['subtotal']   = currency_format_with_sym($value['subtotal']);
                            $mainsubtotal                      += $value['subtotal'];
                        }
                    }
                    $amount = $mainsubtotal;
                    $posPayment->amount         = $amount;
                    $total= $mainsubtotal- $discount;
                    $posPayment->discount         = $discount;
                    $posPayment->discount_amount       = $total;
                    $posPayment->workspace      = getActiveWorkSpace();
                    $posPayment->created_by     = creatorId();
                    $posPayment->save();
                    event(new CreatePaymentPos($request,$posPayment,$pos));
                    session()->forget('pos');
                    return response()->json(
                        [
                            'code' => 200,
                            'success' => __('Payment completed successfully!'),
                        ]
                    );
                }
            } else {
                return response()->json(
                    [
                        'code' => 404,
                        'success' => __('Items not found!'),
                    ]
                );
            }
        } else {
            return redirect()->back()->with('error', __('Permission denied.'));
        }
    }

    function invoicePosNumber()
    {
        if (\Auth::user()->isAbleTo('pos add manage')) {
            $latest = Pos::where('created_by', creatorId())->latest()->first();


            return $latest ? $latest->pos_id + 1 : 1;
        } else {
            return redirect()->back()->with('error', __('Permission denied.'));
        }
    }

    function report(PosOrderDataTable $dataTable)
    {
        if(\Auth::user()->isAbleTo('pos order manage'))
        {
            return $dataTable->render('pos::pos.report');
        }
        else
        {
            return redirect()->back()->with('error', __('Permission Denied.'));
        }

    }
    public function getProductCategories()
    {
        $cat = Pos::getallCategories();


        $all_products = Pos::getallproducts()->count();
        $html = '<div class="zoom-in me-2">
                  <div class="card rounded-1 card-stats mb-0 cat-active overflow-hidden" data-id="0">
                     <div class="category-select" data-cat-id="0">
                        <button type="button" class="btn tab-btns btn-primary">'.__("All Categories").'</button>
                     </div>
                  </div>
               </div>';
        foreach ($cat as $key => $c) {
            $dcls = 'category-select';
            $html .= ' <div class="zoom-in cat-list-btn me-2">
                          <div class="card rounded-1 card-stats mb-0 overflow-hidden " data-id="'.$c->id.'">
                             <div class="'.$dcls.'" data-cat-id="'.$c->id.'">
                                <button type="button" class="btn tab-btns -backdrop-hue-rotate-15 ">'.$c->name.'</button>
                             </div>
                          </div>
                       </div>';
        }
        return Response($html);
    }
    public function searchProducts(Request $request)
    {
        $lastsegment = $request->session_key;
        if (\Auth::user()->isAbleTo('pos add manage') && $request->ajax() && isset($lastsegment) && !empty($lastsegment)) {
            // if(isset($request->cat_id) && $request->cat_id == 0)
            // {
            //     $cart = session()->get($lastsegment);
            //     if (isset($cart) && count($cart) > 0)
            //     {
            //         session()->forget($lastsegment);
            //     }
            // }
            $output = "";
            if($request->war_id == '0')
            {
                $ids = WarehouseProduct::where('warehouse_id',1)->get()->pluck('product_id')->toArray();
                if ($request->cat_id !== '' && $request->search == '')
                {
                    if($request->cat_id == '0')
                    {
                        $products = Pos::getallproducts()->whereIn('product_services.id',$ids)->get();

                    }else{
                        $products = Pos::getallproducts()->where('category_id', $request->cat_id)->whereIn('product_services.id',$ids)->get();
                    }
                } else {
                    if($request->cat_id == '0'){
                        $products = Pos::getallproducts()->where('product_services.name', 'LIKE', "%{$request->search}%")->whereIn('product_services.id',$ids)->get();
                    }else{
                        $products = Pos::getallproducts()->where('product_services.name', 'LIKE', "%{$request->search}%")->orWhere('category_id', $request->cat_id)->get();
                    }

                }
            }else{
                $ids = WarehouseProduct::where('warehouse_id',$request->war_id)->get()->pluck('product_id')->toArray();
                if($request->cat_id == '' || $request->cat_id == '0')
                {
                    $products = Pos::getallproducts()->whereIn('product_services.id',$ids);

                }else{
                    $products = Pos::getallproducts()->where('category_id', $request->cat_id)->whereIn('product_services.id',$ids);
                }

                if(!empty($request->search))
                {
                    $products = $products->where('product_services.name', 'LIKE', "%{$request->search}%");
                }

                $products = $products->get();
            }

            if (count($products)>0)
            {
                foreach ($products as $key => $product)
                {
                    $quantity=$product->warehouseProduct($product->id,$request->war_id!=0?$request->war_id:1);
                    $unit=(!empty($product) && !empty($product->unit()))?$product->unit()->name:'';

                    if(check_file($product->image)){
                        $image_url =$product->image;
                    }else{
                        $image_url =asset('packages/workdo/ProductService/src/Resources/assets/image/img01.jpg');
                    }
                    if ($request->session_key == 'purchases')
                    {
                        $productprice = $product->purchase_price != 0 ? $product->purchase_price : 0;
                    }
                    else if ($request->session_key == 'pos')
                    {
                        $productprice = $product->sale_price != 0 ? $product->sale_price : 0;
                    }
                    else
                    {
                        $productprice = $product->sale_price != 0 ? $product->sale_price : $product->purchase_price;
                    }


                    $output .= '

                            <div class="col-xl-3 col-lg-4 col-sm-6">
                                <div class="tab-pane fade show active toacart w-100" data-url="' . url('add-to-cart/' . $product->id . '/' . $lastsegment .'/'. $request->war_id) .'">
                                    <div class="position-relative card">
                                        <div class="card-image">
                                            <img alt="Image placeholder" src="' . get_file($image_url) . '" class="avatar shadow hover-shadow-lg">
                                            <p class="top-badge badge p-2 badge-danger mb-0">'. $quantity.' '.$unit .'</p>
                                        </div>
                                        <div class="p-0 custom-card-body card-body d-flex ">
                                            <div class="card-body p-3 text-left card-bottom-content">
                                                <h6 class="mb-2 text-dark product-title-name">' . $product->name . '</h6>
                                                <p class="badge bg-primary p-2 mb-0">' . currency_format_with_sym($productprice) . '</p>

                                                                                          </div>
                                        </div>
                                    </div>
                                </div>
                            </div>

                    ';

                }

                return Response($output);
            } else {
                $output='<div class="card card-body col-12 text-center">
                    <h5>'.__("No Product Available").'</h5>
                    </div>';
                return Response($output);
            }
        }
    }

    public function addToCart(Request $request, $id,$session_key,$war)
    {
        if (Auth::user()->isAbleTo('pos cart manage') && $request->ajax()) {
            $product = \Workdo\ProductService\Entities\ProductService::find($id);

            $productquantity = 0;

            if ($product) {
                $productquantity=$product->warehouseProduct($product->id,$war!=0?$war:1);
            }

            if (!$product || ($session_key == 'pos' && $productquantity == 0)) {
                return response()->json(
                    [
                        'code' => 404,
                        'status' => 'Error',
                        'error' => __('This product is out of stock!'),
                    ],
                    404
                );
            }

            $productname = $product->name;

            if ($session_key == 'purchases') {

                $productprice = $product->purchase_price != 0 ? $product->purchase_price : 0;
            } else if ($session_key == 'pos') {

                $productprice = $product->sale_price != 0 ? $product->sale_price : 0;
            } else {

                $productprice = $product->sale_price != 0 ? $product->sale_price : $product->purchase_price;
            }

            $originalquantity = (int)$productquantity;
            $taxes=\Workdo\Pos\Entities\Pos::tax($product->tax_id);
            $totalTaxRate=\Workdo\Pos\Entities\Pos::totalTaxRate($product->tax_id);
            $product_tax='';
            $product_tax_id=[];
            foreach($taxes as $tax){
                $product_tax.=!empty($tax)?"<span class='badge badge-primary'>". $tax->name.' ('.$tax->rate.'%)'."</span><br>":'';
                $product_tax_id[]=$tax->id;
            }

            if(empty($product_tax)){
                $product_tax="-";
            }

            $producttax = $totalTaxRate;


            $tax = ($productprice * $producttax) / 100;

            $subtotal        = $productprice + $tax;
            $cart            = session()->get($session_key);
            $image_url = (check_file($product->image) && get_file($product->image)) ? $product->image : asset('packages/workdo/ProductService/src/Resources/assets/image/img01.jpg');



            $model_delete_id = 'delete-form-' . $id;

            $carthtml = '';
            $carthtml .= '<tr data-product-id="' . $id . '" id="product-id-' . $id . '">
                            <td class="cart-images">

                                <img alt="Image placeholder" src="' .get_file($image_url) . '" class="card-image avatar shadow hover-shadow-lg">

                            </td>

                            <td class="name">' . $productname . '</td>

                            <td class="">
                                   <span class="quantity buttons_added">
                                        <button class="minus">
                                            <svg width="20px" height="20px" viewBox="0 0 32 32" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:sketch="http://www.bohemiancoding.com/sketch/ns">
                                                <g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd" sketch:type="MSPage">
                                                    <g id="Icon-Set-Filled" sketch:type="MSLayerGroup" transform="translate(-518.000000, -1089.000000)" fill="#000000">
                                                        <path d="M540,1106 L528,1106 C527.447,1106 527,1105.55 527,1105 C527,1104.45 527.447,1104 528,1104 L540,1104 C540.553,1104 541,1104.45 541,1105 C541,1105.55 540.553,1106 540,1106 L540,1106 Z M534,1089 C525.163,1089 518,1096.16 518,1105 C518,1113.84 525.163,1121 534,1121 C542.837,1121 550,1113.84 550,1105 C550,1096.16 542.837,1089 534,1089 L534,1089 Z" id="minus-circle" sketch:type="MSShapeGroup">

                                            </path>
                                                    </g>
                                                </g>
                                            </svg>
                                        </button>
                                         <input type="number" step="1" min="1" max="'.$productquantity.'" name="quantity" title="' . __('Quantity') . '" class="input-number" size="4" data-url="' . url('update-cart/') . '" data-id="' . $id . '">
                                        <button class="plus">
                                            <svg width="20px" height="20px" viewBox="0 0 32 32" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:sketch="http://www.bohemiancoding.com/sketch/ns">
                                                <g id="Page-1" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd" sketch:type="MSPage">
                                                    <g id="Icon-Set-Filled" sketch:type="MSLayerGroup" transform="translate(-466.000000, -1089.000000)" fill="#000000">
                                                        <path d="M488,1106 L483,1106 L483,1111 C483,1111.55 482.553,1112 482,1112 C481.447,1112 481,1111.55 481,1111 L481,1106 L476,1106 C475.447,1106 475,1105.55 475,1105 C475,1104.45 475.447,1104 476,1104 L481,1104 L481,1099 C481,1098.45 481.447,1098 482,1098 C482.553,1098 483,1098.45 483,1099 L483,1104 L488,1104 C488.553,1104 489,1104.45 489,1105 C489,1105.55 488.553,1106 488,1106 L488,1106 Z M482,1089 C473.163,1089 466,1096.16 466,1105 C466,1113.84 473.163,1121 482,1121 C490.837,1121 498,1113.84 498,1105 C498,1096.16 490.837,1089 482,1089 L482,1089 Z" id="plus-circle" sketch:type="MSShapeGroup">

                                                        </path>
                                                    </g>
                                                </g>
                                            </svg>
                                        </button>
                                   </span>
                            </td>


                            <td class="tax">' . $product_tax .' </td>

                            <td class="price">' . currency_format_with_sym($productprice) . '</td>

                            <td class="subtotal">' . currency_format_with_sym($subtotal) . '</td>

                            <td class="">
                                 <a href="#" class="action-btn  show_confirm-pos" data-confirm="' . __("Are You Sure?") . '" data-text="' . __("This action can not be undone. Do you want to continue?") . '" data-confirm-yes=' . $model_delete_id . ' title="' . __('Delete') . '}" data-id="' . $id . '" title="' . __('Delete') . '"   >
                                   <span class=""><i class="ti ti-trash bg-danger btn btn-sm text-white"></i></span>
                                 </a>
                                 <form method="post" action="' . url('remove-from-cart') . '"  accept-charset="UTF-8" id="' . $model_delete_id . '">
                                      <input name="_method" type="hidden" value="DELETE">
                                      <input name="_token" type="hidden" value="' . csrf_token() . '">
                                      <input type="hidden" name="session_key" value="' . $session_key . '">
                                      <input type="hidden" name="id" value="' . $id . '">
                                 </form>

                            </td>
                        </td>';


            if (!$cart) {

                $cart = [
                    $id => [
                        "name" => $productname,
                        "quantity" => 1,
                        "price" => $productprice,
                        "id" => $id,
                        "tax" => $producttax,
                        "subtotal" => $subtotal,
                        "originalquantity" => $originalquantity,
                        "product_tax"=>$product_tax,
                        "product_tax_id"=>!empty($product_tax_id)?implode(',',$product_tax_id):0,
                    ],
                ];
                if ($originalquantity < $cart[$id]['quantity'] && $session_key == 'pos') {
                    return response()->json(
                        [
                            'code' => 404,
                            'status' => 'Error',
                            'error' => __('This product is out of stock!'),
                        ],
                        404
                    );
                }

                session()->put($session_key, $cart);

                return response()->json(
                    [
                        'code' => 200,
                        'status' => 'Success',
                        'success' => $productname . __(' added to cart successfully!'),
                        'product' => $cart[$id],
                        'carthtml' => $carthtml,
                    ]
                );
            }

            // if cart not empty then check if this product exist then increment quantity
            if (isset($cart[$id])) {

                $cart[$id]['quantity']++;
                $cart[$id]['id'] = $id;

                $subtotal = $cart[$id]["price"] * $cart[$id]["quantity"];
                $tax      = ($subtotal * $cart[$id]["tax"]) / 100;

                $cart[$id]["subtotal"]         = $subtotal + $tax;
                $cart[$id]["originalquantity"] = $originalquantity;

                if ($originalquantity < $cart[$id]['quantity'] && $session_key == 'pos') {
                    return response()->json(
                        [
                            'code' => 404,
                            'status' => 'Error',
                            'error' => __('This product is out of stock!'),
                        ],
                        404
                    );
                }

                session()->put($session_key, $cart);

                return response()->json(
                    [
                        'code' => 200,
                        'status' => 'Success',
                        'success' => $productname . __(' added to cart successfully!'),
                        'product' => $cart[$id],
                        'carttotal' => $cart,
                    ]
                );
            }

            // if item not exist in cart then add to cart with quantity = 1

            $cart[$id] = [
                "name" => $productname,
                "quantity" => 1,
                "price" => $productprice,
                "tax" => $producttax,
                "subtotal" => $subtotal,
                "id" => $id,
                "originalquantity" => $originalquantity,
                "product_tax"=>$product_tax,
            ];

            if ($originalquantity < $cart[$id]['quantity'] && $session_key == 'pos') {
                return response()->json(
                    [
                        'code' => 404,
                        'status' => 'Error',
                        'error' => __('This product is out of stock!'),
                    ],
                    404
                );
            }

            session()->put($session_key, $cart);

            return response()->json(
                [
                    'code' => 200,
                    'status' => 'Success',
                    'success' => $productname . __(' added to cart successfully!'),
                    'product' => $cart[$id],
                    'carthtml' => $carthtml,
                    'carttotal' => $cart,
                ]
            );
        } else {
            return response()->json(
                [
                    'code' => 404,
                    'status' => 'Error',
                    'error' => __('This Product is not found!'),
                ],
                404
            );
        }
    }
    public function removeFromCart(Request $request)
    {

        $id          = $request->id;
        $session_key = $request->session_key;
        if (Auth::user()->isAbleTo('pos cart manage') && isset($id) && !empty($id) && isset($session_key) && !empty($session_key)) {
            $cart = session()->get($session_key);
            if (isset($cart[$id])) {
                unset($cart[$id]);
                session()->put($session_key, $cart);
            }

            return redirect()->back()->with('error', __('Product removed from cart!'));
        } else {
            return redirect()->back()->with('error', __('This Product is not found!'));
        }
    }

    public function updateCart(Request $request)
    {

        $id          = $request->id;
        $quantity    = $request->quantity;


        $discount    = $request->discount;
        $session_key = $request->session_key;

        if (isset($cart[$id]) && $quantity == 0) {
            unset($cart[$id]);
        }
        if (Auth::user()->isAbleTo('pos cart manage') && $request->ajax() && isset($id) && !empty($id) && isset($session_key) && !empty($session_key))
        {
            $cart = session()->get($session_key);

            if ($quantity) {

                $cart[$id]["quantity"] = $quantity;
                $productprice          = (array_key_exists("price",$cart[$id])) ?  $cart[$id]["price"] : 0;
                $subtotal = $productprice * $quantity;

                $tax = 0;
                if(array_key_exists('tax',$cart[$id]))
                {
                    $producttax            = $cart[$id]["tax"];

                    $tax      = ($subtotal * $producttax) / 100;
                }

                $cart[$id]["subtotal"] = $subtotal + $tax;
            }

            if (isset($cart[$id]["originalquantity"]) && $cart[$id]["originalquantity"] < $cart[$id]['quantity'] && $session_key == 'pos') {
                return response()->json(
                    [
                        'code' => 404,
                        'status' => 'Error',
                        'error' => __('This product is out of stock!'),
                    ],
                    404
                );
            }

            $subtotal = array_sum(array_column($cart, 'subtotal'));
            $discount = (!empty($request->discount) && $request->discount > 0 ) ? $request->discount : 0;
            $total = $subtotal - $discount;
            $totalDiscount = currency_format_with_sym($total);
            $discount = $totalDiscount;

            session()->put($session_key, $cart);

            return response()->json(
                [
                    'code' => 200,
                    'success' => __('The cart are updated successfully!'),
                    'product' => $cart,
                    'discount' => $discount,
                ]
            );
        } else {
            return response()->json(
                [
                    'code' => 404,
                    'status' => 'Error',
                    'error' => __('This Product is not found!'),
                ],
                404
            );
        }
    }

    /**
     * Show the specified resource.
     * @param int $id
     * @return Renderable
     */
    public function show($ids)
    {
        if(\Auth::user()->isAbleTo('pos show'))
        {
            try {
                $id       = Crypt::decrypt($ids);
            } catch (\Throwable $th) {
                return redirect()->back()->with('error', __('POS Not Found.'));
            }
            $pos = Pos::find($id);
            if(isset($pos) && !empty($pos)){

                $company_setting = getCompanyAllSetting();

                if($pos->created_by == creatorId())
                {
                    $posPayment = PosPayment::where('pos_id', $pos->id)->first();
                    $customer=[];
                    if(module_is_active('Account'))
                    {
                        $customer             = $pos->customer;
                    }
                    $iteams               = $pos->itemswithproduct;  //items;
                    return view('pos::pos.view', compact('pos', 'customer','iteams','posPayment','company_setting'));
                }
                else
                {
                    return redirect()->back()->with('error', __('Permission denied.'));
                }
            }
            else
            {
                return redirect()->back()->with('error', __('POS Not Found.'));
            }
        }
        else
        {
            return redirect()->back()->with('error', __('Permission denied.'));
        }
    }

    /**
     * Show the form for editing the specified resource.
     * @param int $id
     * @return Renderable
     */
    public function edit($id)
    {
        return redirect()->back();
    }

    /**
     * Update the specified resource in storage.
     * @param Request $request
     * @param int $id
     * @return Renderable
     */
    public function update(Request $request, $id)
    {
        return redirect()->back();
    }

    /**
     * Remove the specified resource from storage.
     * @param int $id
     * @return Renderable
     */
    public function destroy($id)
    {
        return redirect()->back();
    }

    public function setting(Request $request)
    {
        if(\Auth::user()->isAbleTo('pos setting manage'))
        {
            $validator = Validator::make($request->all(),
            [
                'low_product_stock_threshold'=>'required'
            ]);
            if($validator->fails()){
                $messages = $validator->getMessageBag();
                return redirect()->back()->with('error', $messages->first());
            }
            else
            {
                $post = $request->all();
                unset($post['_token']);
                foreach ($post as $key => $value) {
                    // Define the data to be updated or inserted
                    $data = [
                        'key' => $key,
                        'workspace' => getActiveWorkSpace(),
                        'created_by' =>\Auth::user()->id,
                    ];
                    // Check if the record exists, and update or insert accordingly
                    Setting::updateOrInsert($data, ['value' => $value]);
                }
                // Settings Cache forget
                comapnySettingCacheForget();

                return redirect()->back()->with('success',__('The POS setting has been saved successfully.'));
            }
        }
        else
        {
            return redirect()->back()->with('error', __('Permission denied.'));
        }
    }
    public function emptyCart(Request $request)
    {
        $session_key = $request->session_key;

        if (Auth::user()->isAbleTo('pos cart manage') && isset($session_key) && !empty($session_key))
        {
            $cart = session()->get($session_key);
            if (isset($cart) && count($cart) > 0)
            {
                session()->forget($session_key);
            }

            return redirect()->back()->with('error', __('Cart is empty!'));

        }
        else
        {
            return redirect()->back()->with('error', __('Cart cannot be empty!.'));

        }
    }

    public function warehouseemptyCart(Request $request)
    {
        $session_key = $request->session_key;
            $cart = session()->get($session_key);
            if (isset($cart) && count($cart) > 0)
            {
                session()->forget($session_key);
            }

        return response()->json();

    }

    public function dashboard(Request $request)
    {
        if (\Auth::user()->isAbleTo('pos dashboard manage'))
        {
            $low_stock = company_setting('low_product_stock_threshold');
            $productscount=[];
            $lowstockproducts = [];


            if(module_is_active('ProductService'))
            {
                    $productObj = Pos::getallproducts();
                    $productscount = $productObj->count();

                if ($productscount > 0) {

                    foreach ($productObj as $key => $product) {

                        $productquantity = $product->getTotalProductQuantity();
                        if ($productquantity <= $low_stock) {
                            $lowstockproducts[] = [
                                'name' => $product->name,
                                'quantity' => $productquantity
                            ];
                        }
                    }
                }
            }
            $customers=[];
            $vendors=[];
            if(module_is_active('Account'))
            {
                $customers = \Workdo\Account\Entities\Customer::select('id')->where('created_by', creatorId())->where('workspace',getActiveWorkSpace())->count();

                $vendors = \Workdo\Account\Entities\Vender::select('id')->where('created_by', creatorId())->where('workspace',getActiveWorkSpace())->count();
            }
            $monthlySelledAmount = Pos::totalSelledAmount(true);
            $totalSelledAmount   = Pos::totalSelledAmount();

            $salesArray = Pos::getSalesReportChart();
            $ActiveWorkspaceName = WorkSpace::where('id', getActiveWorkSpace())->pluck('name')->first();
            $homes = [
                'productscount',
                'lowstockproducts',
                'customers',
                'vendors',
                'monthlySelledAmount',
                'totalSelledAmount',
                'salesArray',
                'ActiveWorkspaceName',
            ];
            return view('pos::dashboard.dashboard', compact($homes));
        }
        else
        {
            return redirect()->back()->with('error', __('Permission denied.'));
        }
    }

    public function cartdiscount(Request $request)
    {
        $total = currency_format_with_sym(0);
        if($request->discount){
            $sess = session()->get('pos');
            if(isset($sess) && !empty($sess) && count($sess) > 0){

                $subtotal = !empty($sess)?array_sum(array_column($sess, 'subtotal')):0;
                $discount = (!empty($request->discount) && $request->discount > 0 ) ? $request->discount : 0;
                $total = $subtotal - $discount;
                $total = currency_format_with_sym($total);
            }
        }else{
            $sess = session()->get('pos');
            if(isset($sess) && !empty($sess) && count($sess) > 0){

                $subtotal = !empty($sess)?array_sum(array_column($sess, 'subtotal')):0;
                $discount = 0;
                $total = $subtotal - $discount;
                $total = currency_format_with_sym($total);
            }
        }
        return response()->json(['total' => $total], '200');

    }

    public function pos($pos_id)
    {
        if(\Auth::user()->isAbleTo('pos manage'))
        {
            $posId   = Crypt::decrypt($pos_id);
            $pos  = Pos::where('id', $posId)->first();
            $posPayment = PosPayment::where('pos_id', $pos->id)->first();

            $data  = DB::table('settings');
            $data  = $data->where('id', '=', $pos->created_by);
            $data1 = $data->get();

            $customer = $pos->customer;

            $totalTaxPrice = 0;
            $totalQuantity = 0;
            $totalRate     = 0;
            $totalDiscount = 0;
            $taxesData     = [];
            $items         = [];

            foreach($pos->itemswithproduct as $product) //->items
            {

                $item              = new \stdClass();
                $item->name        = !empty($product->product) ? $product->product->name : '';
                $item->quantity    = $product->quantity;
                $item->tax         = $product->tax;
                $item->discount    = $product->discount;
                $item->price       = $product->price;
                $item->description = $product->description;
                $totalQuantity += $item->quantity;
                $totalRate     += $item->price;
                $totalDiscount += $item->discount;
                $taxes     = Pos::tax($product->tax);
                $itemTaxes = [];
                if(!empty($item->tax))
                {
                    foreach($taxes as $tax)
                    {
                        $taxPrice      = Pos::taxRate($tax->rate, $item->price, $item->quantity);
                        $totalTaxPrice += $taxPrice;

                        $itemTax['name']  = $tax->name;
                        $itemTax['rate']  = $tax->rate . '%';
                        $itemTax['price'] = $taxPrice;
                        $itemTaxes[]      = $itemTax;


                        if(array_key_exists($tax->name, $taxesData))
                        {
                            $taxesData[$tax->name] = $taxesData[$tax->name] + $taxPrice;
                        }
                        else
                        {
                            $taxesData[$tax->name] = $taxPrice;
                        }

                    }

                    $item->itemTax = $itemTaxes;
                }
                else
                {
                    $item->itemTax = [];
                }
                $items[] = $item;
            }

            $pos->itemData      = $items;
            $pos->totalTaxPrice = $totalTaxPrice;
            $pos->totalQuantity = $totalQuantity;
            $pos->totalRate     = $totalRate;
            $pos->totalDiscount = $totalDiscount;
            $pos->taxesData     = $taxesData;

            $company_logo = get_file(sidebar_logo());
            $pos_logo = company_setting('pos_logo');
            if(isset($pos_logo) && !empty($pos_logo))
            {
                $img = get_file($pos_logo);
            }
            else{
                $img = $company_logo;
            }

            $settings['site_rtl'] = company_setting('site_rtl');
            $settings['company_email'] = company_setting('company_email');
            $settings['company_telephone'] = company_setting('company_telephone');
            $settings['company_name'] = company_setting('company_name');
            $settings['company_address'] = company_setting('company_address');
            $settings['company_city'] = company_setting('company_city');
            $settings['company_state'] = company_setting('company_state');
            $settings['company_zipcode'] = company_setting('company_zipcode');
            $settings['company_country'] = company_setting('company_country');
            $settings['registration_number'] = company_setting('registration_number');
            $settings['tax_type'] = company_setting('tax_type');
            $settings['vat_number'] = company_setting('vat_number');
            $settings['pos_footer_title'] = company_setting('pos_footer_title');
            $settings['pos_footer_notes'] = company_setting('pos_footer_notes');
            $settings['pos_shipping_display'] = company_setting('pos_shipping_display');
            $settings['pos_template'] = company_setting('pos_template');
            $settings['pos_color'] = company_setting('pos_color');
            if($pos)
            {

                $color=company_setting('pos_color',$pos->created_by, $pos->workspace);
                if($color){
                    $color=$color;
                }else{
                    $color='ffffff';
                }
                $color      = '#' .$color ;
                $font_color   = PosUtility::getFontColor($color);

                return view('pos::pos.templates.' . $settings['pos_template'], compact('pos','posPayment', 'color', 'settings', 'customer', 'img', 'font_color'));
            }
            else
            {
                return redirect()->back()->with('error', __('Permission denied.'));
            }
        }
        else
        {
            return redirect()->back()->with('error', __('Permission denied.'));
        }
    }

    public function previewPos($template, $color)
    {
        if(\Auth::user()->isAbleTo('pos manage'))
        {
            $objUser  = \Auth::user();

            $pos     = new Pos();
            $posPayment     = new posPayment();
            $customer                   = new \stdClass();
            $customer->email            = '<Email>';
            $customer->shipping_name    = '<Customer Name>';
            $customer->shipping_country = '<Country>';
            $customer->shipping_state   = '<State>';
            $customer->shipping_city    = '<City>';
            $customer->shipping_phone   = '<Customer Phone Number>';
            $customer->shipping_zip     = '<Zip>';
            $customer->shipping_address = '<Address>';
            $customer->billing_name     = '<Customer Name>';
            $customer->billing_country  = '<Country>';
            $customer->billing_state    = '<State>';
            $customer->billing_city     = '<City>';
            $customer->billing_phone    = '<Customer Phone Number>';
            $customer->billing_zip      = '<Zip>';
            $customer->billing_address  = '<Address>';
            $totalTaxPrice = 0;
            $taxesData     = [];
            $items         = [];
            for($i = 1; $i <= 3; $i++)
            {
                $item           = new \stdClass();
                $item->name     = 'Item ' . $i;
                $item->quantity = 1;
                $item->tax      = 5;
                $item->discount = 50;
                $item->price    = 100;
                $item->description    = 'In publishing and graphic design, Lorem ipsum is a placeholder';

                $taxes = [
                    'Tax 1',
                    'Tax 2',
                ];

                $itemTaxes = [];
                foreach($taxes as $k => $tax)
                {
                    $taxPrice         = 10;
                    $totalTaxPrice    += $taxPrice;
                    $itemTax['name']  = 'Tax ' . $k;
                    $itemTax['rate']  = '10 %';
                    $itemTax['price'] = '$10';
                    $itemTaxes[]      = $itemTax;
                    if(array_key_exists('Tax ' . $k, $taxesData))
                    {
                        $taxesData['Tax ' . $k] = $taxesData['Tax 1'] + $taxPrice;
                    }
                    else
                    {
                        $taxesData['Tax ' . $k] = $taxPrice;
                    }
                }
                $item->itemTax = $itemTaxes;
                $items[]       = $item;
            }

            $pos->pos_id    = 1;

            $pos->issue_date = date('Y-m-d H:i:s');
            $pos->itemData   = $items;

            $pos->totalTaxPrice = 60;
            $pos->totalQuantity = 3;
            $pos->totalRate     = 300;
            $pos->totalDiscount = 10;
            $pos->taxesData     = $taxesData;
            $pos->created_by     = creatorId();

            $preview      = 1;
            $color        = '#' . $color;
            $font_color   = User::getFontColor($color);

            $company_logo = get_file(sidebar_logo());
            $pos_logo = company_setting('pos_logo');

            if(isset($pos_logo) && !empty($pos_logo))
            {
                $img = get_file($pos_logo);
            }
            else{
                $img = $company_logo;
            }

            $settings['site_rtl'] = company_setting('site_rtl');
            $settings['company_email'] = company_setting('company_email');
            $settings['company_telephone'] = company_setting('company_telephone');
            $settings['company_name'] = company_setting('company_name');
            $settings['company_address'] = company_setting('company_address');
            $settings['company_city'] = company_setting('company_city');
            $settings['company_state'] = company_setting('company_state');
            $settings['company_zipcode'] = company_setting('company_zipcode');
            $settings['company_country'] = company_setting('company_country');
            $settings['registration_number'] = company_setting('registration_number');
            $settings['tax_type'] = company_setting('tax_type');
            $settings['vat_number'] = company_setting('vat_number');
            $settings['pos_footer_title'] = company_setting('pos_footer_title');
            $settings['pos_footer_notes'] = company_setting('pos_footer_notes');
            $settings['pos_shipping_display'] = company_setting('pos_shipping_display');
            $settings['pos_template'] = company_setting('pos_template');
            $settings['pos_color'] = company_setting('pos_color');

            return view('pos::pos.templates.' . $template, compact('pos', 'preview', 'color', 'img', 'settings', 'customer', 'font_color','posPayment'));
        }
        else
        {
            return redirect()->back()->with('error', __('Permission denied.'));
        }
    }

    public function savePosTemplateSettings(Request $request)
    {
        if(\Auth::user()->isAbleTo('pos manage'))
        {
            $user = \Auth::user();
            $post = $request->all();
            unset($post['_token']);

            if($request->pos_logo)
            {
                $request->validate(
                    [
                        'pos_logo' => 'image|mimes:png',
                    ]
                );

                $pos_logo         = $user->id.'_pos_logo.png';
                $uplaod = upload_file($request,'pos_logo',$pos_logo,'pos_logo');
                if($uplaod['flag'] == 1)
                {
                    $url = $uplaod['url'];
                }
                else{
                    return redirect()->back()->with('error',$uplaod['msg']);
                }
            }
            if (isset($request->pos_footer_notes) && !empty($request->pos_footer_notes)){
                $validator = Validator::make($request->all(),
                    [
                        'pos_footer_notes' => 'required|string|regex:/^[^\r\n]*$/',
                    ]
                );
                if ($validator->fails()) {
                    $messages = $validator->getMessageBag();
                    return redirect()->back()->with('error', $messages->first());
                }
            }

            if(isset($post['pos_template']) && (!isset($post['pos_color']) || empty($post['pos_color'])))
            {
                $post['pos_color'] = "ffffff";
            }
            if(isset($post['pos_logo']))
            {
                $post['pos_logo'] = $url;
            }
            if(!isset($post['pos_shipping_display']))
            {
                $post['pos_shipping_display'] = 'off';
            }
            foreach ($post as $key => $value) {
                // Define the data to be updated or inserted
                $data = [
                    'key' => $key,
                    'workspace' => getActiveWorkSpace(),
                    'created_by' =>\Auth::user()->id,
                ];
                // Check if the record exists, and update or insert accordingly
                Setting::updateOrInsert($data, ['value' => $value]);
            }
            // Settings Cache forget
            comapnySettingCacheForget();

            return redirect()->back()->with('success', __('The POS Setting are updated successfully'));
        } else {
            return redirect()->back()->with('error', __('Permission Denied.'));
        }
    }

    public function grid()
    {
        if(\Auth::user()->isAbleTo('pos order manage'))
        {
            $posPayments = Pos::where('created_by', creatorId())->where('workspace',getActiveWorkSpace())->with('customer')->orderBy('id','desc');
            $posPayments = $posPayments->paginate(11);
            return view('pos::pos.grid',compact('posPayments'));
        }
        else
        {
            return redirect()->back()->with('error', __('Permission Denied.'));
        }

    }

    //for thermal print

    public function printView(Request $request)
    {
        if(\Auth::user()->isAbleTo('pos manage'))
        {
            $user = Auth::user();

            if(module_is_active('Account'))
                {
                    $user =  User::where('name', '=', $request->vc_name)->where('type','client')->where('created_by', creatorId())->where('workspace_id',getActiveWorkSpace())->first();
                    if(!empty($user->id))
                    {

                        $customer = \Workdo\Account\Entities\Customer::where('user_id',$user->id)->where('name', '=', $request->vc_name)->where('created_by', creatorId())->where('workspace',getActiveWorkSpace())->first();
                        if(!empty($customer)){
                            $customer = $customer;
                        }
                        else{
                            $customer = $user;
                        }
                    }
                    else{
                        $customer = NULL;
                        $user = Auth::user();
                    }

                }
                else
                {
                    $customer      = User::where('name', '=', $request->vc_name)->where('type','client')->where('created_by', creatorId())->where('workspace_id',getActiveWorkSpace())->first();

                }
            
            $warehouse = warehouse::where('id', '=', $request->warehouse_name)->where('created_by', creatorId())->first();

            $details = [
                'pos_id' =>Pos::posNumberFormat($this->invoicePosNumber()),
                'customer' => $customer != null ? $customer->toArray() : [],
                'warehouse' => $warehouse != null ? $warehouse->toArray() : [],
                'user' => $user != null ? $user->toArray() : [],
                'date' => date('Y-m-d'),
                'pay' => 'show',
            ];

            if (!empty($details['customer']['billing_state']))
            {
                $warehousedetails = '<h7 class="text-dark">' . ucfirst($details['warehouse']['name'])  . '</p></h7>';
                $details['customer']['billing_state'] = $details['customer']['billing_state'] != '' ? ", " . $details['customer']['billing_state'] : '';
                $details['customer']['shipping_state'] = $details['customer']['shipping_state'] != '' ? ", " . $details['customer']['shipping_state'] : '';
                $customerdetails = '<h6 class="text-dark">' . ucfirst($details['customer']['name']) . '<p class="m-0 h6 font-weight-normal">' . $details['customer']['billing_phone'] . '</p>' . '<p class="m-0 h6 font-weight-normal">' . $details['customer']['billing_address'] . '</p>' . '<p class="m-0 h6 font-weight-normal">' . $details['customer']['billing_city'] . $details['customer']['billing_state'] . '</p>' . '<p class="m-0 h6 font-weight-normal">' . $details['customer']['billing_country'] . '</p>' . '<p class="m-0 h6 font-weight-normal">' . $details['customer']['billing_zip'] . '</p></h6>';
                $shippdetails = '<h6 class="text-dark"><b>' . ucfirst($details['customer']['name']) . '</b>' . '<p class="m-0 h6 font-weight-normal">' . $details['customer']['shipping_phone'] . '</p>' . '<p class="m-0 h6 font-weight-normal">' . $details['customer']['shipping_address'] . '</p>' . '<p class="m-0 h6 font-weight-normal">' . $details['customer']['shipping_city'] . $details['customer']['shipping_state'] . '</p>' . '<p class="m-0 h6 font-weight-normal">' . $details['customer']['shipping_country'] . '</p>' . '<p class="m-0 h6 font-weight-normal">' . $details['customer']['shipping_zip'] . '</p></h6>';

            }
            else {
                $customerdetails = '<h2 class="h6"><b>' . __('Walk-in Customer') . '</b><h2>';
                $warehousedetails = '<h7 class="text-dark">' . ucfirst($details['warehouse']['name'])  . '</p></h7>';
                $shippdetails = '-';

            }

            $settings['company_telephone'] = company_setting('company_telephone') != '' ? ", " . company_setting('company_telephone') : '';
            $settings['company_state']     = company_setting('company_state') != '' ? ", " . company_setting('company_state') : '';

            $userdetails = '<h6 class="text-dark"><b>' . ucfirst($details['user']['name']) . ' </b> <h2  class="font-weight-normal">' . '<p class="m-0 font-weight-normal">' . company_setting('company_name') . company_setting('company_telephone') . '</p>' . '<p class="m-0 font-weight-normal">' . company_setting('company_address') . '</p>' . '<p class="m-0 h6 font-weight-normal">' . company_setting('company_city') . company_setting('company_state') . '</p>' . '<p class="m-0 font-weight-normal">' . company_setting('company_country') . '</p>' . '<p class="m-0 font-weight-normal">' . company_setting('company_zipcode') . '</p></h2>';

            $details['customer']['details'] = $customerdetails;
            $details['warehouse']['details'] = $warehousedetails;

            $details['customer']['shippdetails'] = $shippdetails;

            $details['user']['details'] = $userdetails;

            $mainsubtotal = 0;
            $sales        = [];
            $sess = session()->get('pos');
            if(isset($sess) && !empty($sess) && count($sess) > 0){
                foreach ($sess as $key => $value) {

                    $subtotal = $value['price'] * $value['quantity'];
                    $tax      = ($subtotal * $value['tax']) / 100;
                    $sales['data'][$key]['name']       = $value['name'];
                    $sales['data'][$key]['quantity']   = $value['quantity'];
                    $sales['data'][$key]['price']      = currency_format_with_sym($value['price']);
                    $sales['data'][$key]['tax']        = $value['tax'] . '%';
                    $sales['data'][$key]['product_tax']        = $value['product_tax'];
                    $sales['data'][$key]['tax_amount'] = currency_format_with_sym($tax);
                    $sales['data'][$key]['subtotal']   = currency_format_with_sym($value['subtotal']);
                    $mainsubtotal                      += $value['subtotal'];
                }
            }

            if($request->discount <= $mainsubtotal){
                $discount=!empty($request->discount)?$request->discount:0;
            }
            else{
                $discount=$mainsubtotal;
            }
            $sales['discount'] = currency_format_with_sym($discount);
            $total= $mainsubtotal-$discount;
            $sales['sub_total'] = currency_format_with_sym($mainsubtotal);
            $sales['total'] = currency_format_with_sym($total);

            return view('pos::pos.printview', compact('details', 'sales', 'customer'));
        } else {
            return redirect()->back()->with('error', __('Permission Denied.'));
        }

    }


    public function barcode(BarcodeDataTable $dataTable)
    {
        if (Auth::user()->isAbleTo('print_barcode manage')) {
            if(module_is_active('ProductService'))
            {
                $barcode = [
                    'barcodeType' => Pos::barcodeType(),
                    'barcodeFormat' => Pos::barcodeFormat(),
                ];
                return $dataTable->render('pos::barcode.barcode',compact('barcode'));
            }else{
                return redirect()->back()->with('error', __('Please Enable Product & Service Module.'));
            }
        } else {
            return redirect()->back()->with('error', __('Permission Denied.'));
        }
    }

    public function barcodeSetting()
    {
        if (Auth::user()->isAbleTo('print_barcode manage')) {
            $settings = getCompanyAllSetting();

            return view('pos::barcode.setting', compact('settings'));
        } else {
            return response()->json(['error' => __('Permission denied.')], 401);
        }
    }

    public function BarcodesettingStore(Request $request)
    {
        if (Auth::user()->isAbleTo('print_barcode manage')) {
            $post['barcode_type'] = isset($request->barcode_type) ? $request->barcode_type : 'code128';
            $post['barcode_format'] = isset($request->barcode_format) ? $request->barcode_format : 'css';

            foreach ($post as $key => $value) {
                // Define the data to be updated or inserted
                $data = [
                    'key' => $key,
                    'workspace' => getActiveWorkSpace(),
                    'created_by' =>\Auth::user()->id,
                ];
                // Check if the record exists, and update or insert accordingly
                Setting::updateOrInsert($data, ['value' => $value]);
            }
            // Settings Cache forget
            comapnySettingCacheForget();

            return redirect()->back()->with('success', __('The barcode setting are updated successfully.'));
        } else {
            return redirect()->back()->with('error', __('Permission Denied.'));
        }
    }

    public function printBarcode()
    {
        if (Auth::user()->isAbleTo('print_barcode manage')) {
            $warehouses = warehouse::select('*', \DB::raw("CONCAT(name) AS name"))->where('created_by', creatorId())->where('workspace',getActiveWorkSpace())->get()->pluck('name', 'id');

            return view('pos::barcode.print', compact('warehouses'));
        } else {
            return redirect()->back()->with('error', __('Permission Denied.'));
        }
    }

    public function getproduct(Request $request)
    {
        if ($request->warehouse_id == 0) {
            $productServices = WarehouseProduct::where('product_id', '=', $request->warehouse_id)->where('created_by', '=', creatorId())->where('workspace',getActiveWorkSpace())->get()->pluck('name', 'id')->toArray();
        } else {
            $productServicesId = WarehouseProduct::where('created_by', '=', creatorId())->where('warehouse_id', $request->warehouse_id)->where('workspace',getActiveWorkSpace())->get()->pluck('product_id')->toArray();
            if(module_is_active('ProductService'))
            {
                $productServices = \Workdo\ProductService\Entities\ProductService::whereIn('id', $productServicesId)->where('created_by', creatorId())->where('workspace_id',getActiveWorkSpace())->get()->pluck('name', 'id')->toArray();
            }else{
                $productServices = [];
            }
        }
        return response()->json($productServices);
    }

    public function receipt(Request $request)
    {
        if (Auth::user()->isAbleTo('print_barcode manage')) {
            if (!empty($request->product_id)) {
                if(module_is_active('ProductService'))
                {
                    $productServices = \Workdo\ProductService\Entities\ProductService::whereIn('id', $request->product_id)->where('created_by', creatorId())->where('workspace_id',getActiveWorkSpace())->get();
                }else{
                    $productServices = [];
                }
                $quantity = $request->quantity;
                $barcode = [
                    'barcodeType' => Pos::barcodeType() == '' ? 'code128' : Pos::barcodeType(),
                    'barcodeFormat' => Pos::barcodeFormat() == '' ? 'css' : Pos::barcodeFormat(),
                ];
            } else {
                return redirect()->back()->with('error', __('Product is required.'));
            }
            return view('pos::barcode.receipt', compact('productServices', 'barcode', 'quantity'));
        } else {
            return redirect()->back()->with('error', __('Permission Denied.'));
        }
    }

}
