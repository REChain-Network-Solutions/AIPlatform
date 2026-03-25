<div class="modal-dialog" role="document">
  <div class="modal-content">

    {!! Form::open(['url' => action([\Modules\Accounting\Http\Controllers\TransferController::class, 'store']), 
        'method' => 'post', 'id' => 'transfer_form' ]) !!}

    <div class="modal-header">
      <button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">&times;</span></button>
      <h4 class="modal-title">@lang( 'accounting::lang.add_transfer' )</h4>
    </div>

    <div class="modal-body">
        <div class="form-group">
            {!! Form::label('ref_no', __('purchase.ref_no').':') !!}
            @show_tooltip(__('lang_v1.leave_empty_to_autogenerate'))
            {!! Form::text('ref_no', null, ['class' => 'form-control']); !!}
        </div>
        <div class="form-group">
            {!! Form::label('from_account', __( 'lang_v1.transfer_from' ) .":*") !!}
            {!! Form::select('from_account', [], null, ['class' => 'form-control accounts-dropdown', 'required', 
                'placeholder' => __('messages.please_select') ]); !!}
        </div>

        <div class="form-group">
            {!! Form::label('to_account', __( 'account.transfer_to' ) .":*") !!}
            {!! Form::select('to_account', [], null, ['class' => 'form-control accounts-dropdown', 'required', 
                'placeholder' => __('messages.please_select') ]); !!}
        </div>

        <div class="form-group">
            {!! Form::label('amount', __( 'sale.amount' ) .":*") !!}
            {!! Form::text('amount', 0, ['class' => 'form-control input_number', 
                'required','placeholder' => __( 'sale.amount' ) ]); !!}
        </div>

        <div class="form-group">
            {!! Form::label('operation_date', __( 'messages.date' ) .":*") !!}
            <div class="input-group">
                {!! Form::text('operation_date', null, ['class' => 'form-control', 
                    'required','placeholder' => __( 'messages.date' ), 'id' => 'operation_date' ]); !!}
                <span class="input-group-addon">
                <span class="glyphicon glyphicon-calendar"></span>
                </span>
            </div>
        </div>

        <div class="form-group">
            {!! Form::label('note', __( 'brand.note' )) !!}
            {!! Form::textarea('note', null, ['class' => 'form-control', 
                'placeholder' => __( 'brand.note' ), 'rows' => 4]); !!}
        </div>
    </div>

    <div class="modal-footer">
      <button type="submit" class="tw-dw-btn tw-dw-btn-primary tw-text-white">@lang( 'messages.save' )</button>
      <button type="button" class="tw-dw-btn tw-dw-btn-neutral tw-text-white" data-dismiss="modal">@lang( 'messages.close' )</button>
    </div>

    {!! Form::close() !!}

  </div><!-- /.modal-content -->
</div><!-- /.modal-dialog -->