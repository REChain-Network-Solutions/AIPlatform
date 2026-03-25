<form class="" method="post" action="{{ route('projects.share',[$project->id]) }}">
    @csrf
    <div class="modal-body">
        <div class="form-group col-md-12 mb-0">
            <label for="users_list" class="col-form-label">{{ __('Clients') }}</label>
            <select class="multi-select choices" id="clients" data-toggle="select2" required name="clients[]" multiple="multiple" data-placeholder="{{ __('Select Clients ...') }}">
                @foreach($clients as $client)
                    <option value="{{$client->id}}">{{$client->name}} - {{$client->email}}</option>
                @endforeach
            </select>
            <p class="text-danger d-none" id="clients_validation">{{__('Clients filed is required.')}}</p>
        </div>
    </div>
    <div class="modal-footer">
           <button type="button" class="btn  btn-light" data-bs-dismiss="modal">{{ __('Cancel')}}</button>
             <input type="submit" value="{{ __('Share to Client')}}" id="submit" class="btn  btn-primary">
        </div>
</form>

<script>
    $(function(){
        $("#submit").click(function() {
            var client =  $("#clients option:selected").length;
            if(client == 0){
            $('#clients_validation').removeClass('d-none')
                return false;
            }else{
            $('#clients_validation').addClass('d-none')
            }
        });
    });
</script>
