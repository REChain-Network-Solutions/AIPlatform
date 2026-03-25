<div class="modal-dialog modal-lg" role="document">
  <div class="modal-content">

    {!! Form::open(['url' => action([\Modules\Accounting\Http\Controllers\AccountTypeController::class, 'update'], 
        $account_type->id), 'method' => 'put', 'id' => 'edit_account_type_form' ]) !!}
    <div class="modal-header">
      <button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">&times;</span></button>
      <h4 class="modal-title">@if($account_type->account_type=='sub_type') 
        @lang('accounting::lang.edit_account_type') @else @lang('accounting::lang.edit_detail_type') @endif</h4>
    </div>

    <div class="modal-body">
        <div class="row">
            <div class="col-md-12">
                <div class="form-group">
                    {!! Form::label('edit_name', __( 'user.name' ) . ':*') !!}
                    {!! Form::text('name', $account_type->name, ['class' => 'form-control', 
                        'required', 'placeholder' => __( 'user.name' ), 'id' => 'edit_name' ]); !!}
                </div>
            </div>
        </div>
        <div class="row @if($account_type->account_type=='sub_type') hide @endif">
            <div class="col-md-12">
              <div class="form-group">
                {!! Form::label('edit_parent_id', __( 'accounting::lang.parent_type' ) . ':*') !!}
                  <select class="form-control" style="width: 100%;" name="parent_id" id="edit_parent_id">
                   <option value="">@lang('messages.please_select')</option>
                     @foreach($account_sub_types as $at)
                       <option value="{{$at->id}}" @if($at->id==$account_type->parent_id) selected @endif >
                      {{$at->account_type_name}}</option>
                     @endforeach
                  </select>
              </div>
            </div>
        </div>
        <div class="row @if($account_type->account_type=='sub_type') hide @endif"">
            <div class="col-md-12">
                <div class="form-group">
                    {!! Form::label('edit_description', __( 'lang_v1.description' ) . ':') !!}
                    {!! Form::textarea('description', $account_type->description, ['class' => 'form-control', 
                        'placeholder' => __( 'lang_v1.description' ), 'rows' => 3, 'id' => 'edit_description' ]); !!}
                </div>
            </div>
        </div> 
    </div>

    <div class="modal-footer">
      <button type="submit" class="tw-dw-btn tw-dw-btn-primary tw-text-white">@lang( 'messages.update' )</button>
      <button type="button" class="tw-dw-btn tw-dw-btn-neutral tw-text-white" data-dismiss="modal">@lang( 'messages.close' )</button>
    </div>

    {!! Form::close() !!}

  </div><!-- /.modal-content -->
</div><!-- /.modal-dialog -->