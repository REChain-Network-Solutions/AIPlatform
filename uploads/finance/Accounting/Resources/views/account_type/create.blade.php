<div class="modal fade" id="create_account_type_modal" tabindex="-1" role="dialog">
<div class="modal-dialog modal-lg" role="document">
  <div class="modal-content">

    {!! Form::open(['url' => action([\Modules\Accounting\Http\Controllers\AccountTypeController::class, 'store']), 
        'method' => 'post', 'id' => 'create_account_type_form' ]) !!}
    {!! Form::hidden('account_type', null, ['id' => 'account_type']); !!}
    <div class="modal-header">
      <button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">&times;</span></button>
      <h4 class="modal-title" id="account_type_title"></h4>
    </div>

    <div class="modal-body">
        <div class="row">
            <div class="col-md-12">
                <div class="form-group">
                    {!! Form::label('name', __( 'user.name' ) . ':*') !!}
                    {!! Form::text('name', null, ['class' => 'form-control', 'required', 'placeholder' => __( 'user.name' ) ]); !!}
                </div>
            </div>
        </div>

        <div class="row" id="account_type_div">
            <div class="col-md-12">
              <div class="form-group">
                {!! Form::label('parent_id', __( 'accounting::lang.account_type' ) . ':*') !!}
                  <select class="form-control" style="width: 100%;" name="account_primary_type" id="account_primary_type">
                   <option value="">@lang('messages.please_select')</option>
                        @foreach($account_types as $k => $v)
                            <option value="{{$k}}">{{$v['label']}}</option>
                        @endforeach
                  </select>
              </div>
            </div>
        </div>

        <div class="row" id="parent_id_div">
            <div class="col-md-12">
              <div class="form-group">
                {!! Form::label('parent_id', __( 'accounting::lang.parent_type' ) . ':*') !!}
                  <select class="form-control" style="width: 100%;" name="parent_id" id="parent_id">
                   <option value="">@lang('messages.please_select')</option>
                     @foreach($account_sub_types as $account_type)
                       <option value="{{$account_type->id}}">
                      {{$account_type->account_type_name}}</option>
                     @endforeach
                  </select>
              </div>
            </div>
        </div>
        <div class="row" id="description_div">
            <div class="col-md-12">
                <div class="form-group">
                    {!! Form::label('description', __( 'lang_v1.description' ) . ':') !!}
                    {!! Form::textarea('description', null, ['class' => 'form-control', 
                        'placeholder' => __( 'lang_v1.description' ), 'rows' => 3 ]); !!}
                </div>
            </div>
        </div> 
    </div>

    <div class="modal-footer">
      <button type="submit" class="tw-dw-btn tw-dw-btn-primary tw-text-white">@lang( 'messages.save' )</button>
      <button type="button" class="tw-dw-btn tw-dw-btn-neutral tw-text-white" data-dismiss="modal">@lang( 'messages.close' )</button>
    </div>

    {!! Form::close() !!}

  </div><!-- /.modal-content -->
</div><!-- /.modal-dialog -->
</div>