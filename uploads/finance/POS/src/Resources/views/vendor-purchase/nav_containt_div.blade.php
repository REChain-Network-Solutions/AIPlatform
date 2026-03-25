<div class="tab-pane fade" id="vendor-purchase" role="tabpanel" aria-labelledby="pills-user-tab-4">
    <div class="row">
        <div class="col-sm-12">
            <div class="card">
                <div class="card-body table-border-style">
                    <div class="table-responsive">
                        <table class="table mb-0 pc-dt-simple" id="vendor_purchase">
                            <thead>
                                <tr>
                                    <th>{{__('Purchase')}}</th>
                                    <th>{{__('Purchase Date')}}</th>
                                    <th>{{__('Due Amount')}}</th>
                                    <th>{{__('Status')}}</th>
                                    @if(Laratrust::hasPermission('purchase edit') || Laratrust::hasPermission('purchase delete') || Laratrust::hasPermission('purchase show'))
                                        <th width="10%"> {{__('Action')}}</th>
                                    @endif
                                </tr>
                            </thead>
                            <tbody>
                                @forelse(App\Models\Purchase::vendorPurchase($vendor->id) as $purchase)
                                <tr class="font-style">
                                    <td class="Id">
                                        @permission('purchase show')
                                            <a href="{{ route('purchases.show',\Crypt::encrypt($purchase->id)) }}" class="btn btn-outline-primary">{{ App\Models\Purchase::purchaseNumberFormat($purchase->purchase_id) }}</a>
                                        @else
                                            <a  class="btn btn-outline-primary">{{ App\Models\Purchase::purchaseNumberFormat($purchase->purchase_id) }}</a>
                                        @endpermission
                                    </td>
                                    <td>{{ company_date_formate($purchase->purchase_date) }}</td>
                                    <td>{{ currency_format_with_sym($purchase->getDue())  }}</td>
                                    <td>
                                        @if($purchase->status == 0)
                                            <span class="badge bg-primary p-2 px-3">{{ __(App\Models\Purchase::$statues[$purchase->status]) }}</span>
                                        @elseif($purchase->status == 1)
                                            <span class="badge bg-warning p-2 px-3">{{ __(App\Models\Purchase::$statues[$purchase->status]) }}</span>
                                        @elseif($purchase->status == 2)
                                            <span class="badge bg-danger p-2 px-3">{{ __(App\Models\Purchase::$statues[$purchase->status]) }}</span>
                                        @elseif($purchase->status == 3)
                                            <span class="badge bg-info p-2 px-3">{{ __(App\Models\Purchase::$statues[$purchase->status]) }}</span>
                                        @elseif($purchase->status == 4)
                                            <span class="badge bg-success p-2 px-3">{{ __(App\Models\Purchase::$statues[$purchase->status]) }}</span>
                                        @endif
                                    </td>
                                    @if(Laratrust::hasPermission('purchase edit') || Laratrust::hasPermission('purchase delete') || Laratrust::hasPermission('purchase show'))
                                    <td class="Action">
                                        <span>
                                            @permission('purchase show')
                                                <div class="action-btn me-2">
                                                    <a href="{{ route('purchases.show',\Crypt::encrypt($purchase->id)) }}" class="mx-3 btn btn-sm align-items-center bg-warning" data-bs-toggle="tooltip" title="{{__('Show')}}" data-original-title="{{__('Detail')}}">
                                                        <i class="ti ti-eye text-white"></i>
                                                    </a>
                                                </div>
                                            @endpermission
                                            @permission('purchase edit')
                                                <div class="action-btn me-2">
                                                    <a href="{{ route('purchases.edit',\Crypt::encrypt($purchase->id)) }}" class="mx-3 btn btn-sm align-items-center bg-info" data-bs-toggle="tooltip" title="Edit" data-original-title="{{__('Edit')}}">
                                                        <i class="ti ti-pencil text-white"></i>
                                                    </a>
                                                </div>
                                            @endpermission
                                            @permission('purchase delete')
                                                <div class="action-btn">
                                                    {!! Form::open(['method' => 'DELETE', 'route' => ['purchases.destroy', $purchase->id],'class'=>'delete-form-btn','id'=>'delete-form-'.$purchase->id]) !!}
                                                        <a href="#" class="mx-3 btn btn-sm align-items-center show_confirm bg-danger"  data-bs-toggle="tooltip" title="{{__('Delete')}}" data-original-title="{{__('Delete')}}" data-confirm="{{__('Are You Sure?')}}" data-confirm-yes="document.getElementById('delete-form-{{$purchase->id}}').submit();">
                                                            <i class="ti ti-trash text-white"></i>
                                                        </a>
                                                    {!! Form::close() !!}
                                                </div>
                                            @endpermission
                                        </span>
                                    </td>
                                @endif
                                </tr>
                                @empty
                                    @include('layouts.nodatafound')
                            @endforelse
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
