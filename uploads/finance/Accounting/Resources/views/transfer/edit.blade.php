<div class="modal-dialog" role="document">
  <div class="modal-content">

    {!! Form::open(['url' => action([\Modules\Accounting\Http\Controllers\TransferController::class, 'update'], 
        $mapping_transaction->id), 'method' => 'put', 'id' => 'transfer_form' ]) !!}

    <div class="modal-header">
      <button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">&times;</span></button>
      <h4 class="modal-title">@lang( 'accounting::lang.edit_transfer' )</h4>
    </div>

    <div class="modal-body">
        <div class="form-group">
            {!! Form::label('ref_no', __('purchase.ref_no').':') !!}
            @show_tooltip(__('lang_v1.leave_empty_to_autogenerate'))
            {!! Form::text('ref_no', $mapping_transaction->ref_no, ['class' => 'form-control']); !!}
        </div>
        <div class="form-group">
            {!! Form::label('from_account', __( 'lang_v1.transfer_from' ) .":*") !!}
            {!! Form::select('from_account', [], $debit_tansaction->accounting_account_id, ['class' => 'form-control', 'required', 
                'placeholder' => __('messages.please_select') ]); !!}
        </div>

        <div class="form-group">
            {!! Form::label('to_account', __( 'account.transfer_to' ) .":*") !!}
            {!! Form::select('to_account', [], $credit_tansaction->accounting_account_id, ['class' => 'form-control', 'required', 
                'placeholder' => __('messages.please_select') ]); !!}
        </div>

        <div class="form-group">
            {!! Form::label('amount', __( 'sale.amount' ) .":*") !!}
            {!! Form::text('amount', @num_format($debit_tansaction->amount), ['class' => 'form-control input_number', 
                'required','placeholder' => __( 'sale.amount' ) ]); !!}
        </div>

        <div class="form-group">
            {!! Form::label('operation_date', __( 'messages.date' ) .":*") !!}
            <div class="input-group">
                {!! Form::text('operation_date', @format_datetime($mapping_transaction->operation_date), ['class' => 'form-control', 
                    'required','placeholder' => __( 'messages.date' ), 'id' => 'operation_date' ]); !!}
                <span class="input-group-addon">
                <span class="glyphicon glyphicon-calendar"></span>
                </span>
            </div>
        </div>

        <div class="form-group">
            {!! Form::label('note', __( 'brand.note' )) !!}
            {!! Form::textarea('note', $mapping_transaction->note, ['class' => 'form-control', 
                'placeholder' => __( 'brand.note' ), 'rows' => 4]); !!}
        </div>
    </div>

    <div class="modal-footer">
      <button type="submit" class="tw-dw-btn tw-dw-btn-primary tw-text-white">@lang( 'messages.update' )</button>
      <button type="button" class="tw-dw-btn tw-dw-btn-neutral tw-text-white" data-dismiss="modal">@lang( 'messages.close' )</button>
    </div>

    {!! Form::close() !!}

  </div><!-- /.modal-content -->
</div><!-- /.modal-dialog -->