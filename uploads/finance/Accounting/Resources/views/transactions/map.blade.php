<div class="modal-dialog no-print" role="document">
{!! Form::open(['url' => action([\Modules\Accounting\Http\Controllers\TransactionController::class, 'saveMap']), 'method' => 'POST', 'id' => 'save_accounting_map' ]) !!}
    
    <input type="hidden" name="type" value="{{$type}}" id="transaction_type">
    @if(in_array($type, ['sell', 'purchase', 'expense']))
        <input type="hidden" name="id" value="{{$transaction->id}}">
    @elseif(in_array($type, ['sell_payment', 'purchase_payment']))
        <input type="hidden" name="id" value="{{$transaction_payment->id}}">
    @endif

<div class="modal-content">
    <div class="modal-header">
    <button type="button" class="close no-print" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">&times;</span></button>
    <h4 class="modal-title" id="modalTitle">
        @if($type == 'sell')
            {{$transaction->invoice_no}}
        @elseif(in_array($type, ['sell_payment', 'purchase_payment']))
            {{$transaction_payment->payment_ref_no}}
        @elseif($type == 'purchase' || $type == 'expense')
            {{$transaction->ref_no}}
        @endif
    </h4>
</div>
<div class="modal-body">
    <div class="row">
        <div class="col-md-6">
            <div class="form-group">
                {!! Form::label('payment_account', __('accounting::lang.payment_account') . ':*' ) !!}
                {!! Form::select('payment_account', !is_null($default_payment_account) ? [$default_payment_account->id => $default_payment_account->name] : [], $default_payment_account->id ?? null, ['class' => 'form-control accounts-dropdown','placeholder' => __('accounting::lang.payment_account'), 'required' => 'required']); !!}
            </div>
        </div>

        <div class="col-md-6">
            <div class="form-group">
                {!! Form::label('deposit_to', __('accounting::lang.deposit_to') . ':*' ) !!}
                {!! Form::select('deposit_to', !is_null($default_deposit_to) ? 
                    [$default_deposit_to->id => $default_deposit_to->name] : [], $default_deposit_to->id ?? null, ['class' => 'form-control accounts-dropdown','placeholder' => __('accounting::lang.deposit_to'), 'required' => 'required']); !!}
            </div>
        </div>
        <div class="col-md-12">
                <div class="form-group">
                    {!! Form::label('description', __( 'lang_v1.description' ) . ':') !!}
                    {!! Form::textarea('description', $note, ['class' => 'form-control', 
                        'placeholder' => __( 'lang_v1.description' ), 'rows' => 3 ]); !!}
                </div>
            </div>
    </div>

</div>

<div class="modal-footer">
    <button type="submit" class="tw-dw-btn tw-dw-btn-primary tw-text-white">@lang('messages.update')</button>
    <button type="button" class="tw-dw-btn tw-dw-btn-neutral tw-text-white" data-dismiss="modal">@lang('messages.cancel')</button>
</div>

{!! Form::close() !!}
	</div><!-- /.modal-content -->
</div><!-- /.modal-dialog -->