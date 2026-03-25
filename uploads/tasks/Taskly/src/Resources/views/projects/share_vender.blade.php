<form class="" method="post" action="{{ route('projects.share.vender',[$project->id]) }}">
    @csrf
    <div class="modal-body">
        <div class="form-group col-md-12 mb-0">
            <label for="users_list" class="col-form-label">{{ __('vendors') }}</label>
            <select class="multi-select choices" id="venders" data-toggle="select2" required name="vendors[]" multiple="multiple" data-placeholder="{{ __('Select vendors ...') }}">
                @foreach($venders as $client)
                    <option value="{{$client->id}}">{{$client->name}} - {{$client->email}}</option>
                @endforeach
            </select>
            <p class="text-danger d-none" id="venders_validation">{{__('vendors filed is required.')}}</p>
        </div>
    </div>
    <div class="modal-footer">
           <button type="button" class="btn  btn-light" data-bs-dismiss="modal">{{ __('Cancel')}}</button>
             <input type="submit" value="{{ __('Share to Vendor')}}" id="submit" class="btn  btn-primary">
        </div>
</form>

<script>
    $(function(){
        $("#submit").click(function() {
            var client =  $("#venders option:selected").length;
            if(client == 0){
            $('#venders_validation').removeClass('d-none')
                return false;
            }else{
            $('#venders_validation').addClass('d-none')
            }
        });
    });
</script>
