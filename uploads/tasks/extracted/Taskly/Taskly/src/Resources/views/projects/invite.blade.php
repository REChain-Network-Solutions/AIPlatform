<form class="" method="post" action="{{ route('projects.invite.update',[$project->id]) }}">
    @csrf
    <div class="modal-body">
        <div class="form-group col-md-12 ">
            <label for="users_list" class="form-label">{{ __('Users') }}</label>
            <select class=" multi-select choices" required id="users_list" name="users_list[]"  multiple="multiple" data-placeholder="{{ __('Select Users ...') }}">
                @foreach($workspace_users as $user)
                        <option value="{{$user->email}}">{{$user->name}} - {{$user->email}}</option>
                @endforeach
            </select>
            <p class="text-danger d-none" id="user_validation">{{__('Users filed is required.')}}</p>
        </div>
    </div>
    <div class="modal-footer">
        <button type="button" class="btn  btn-light" data-bs-dismiss="modal">{{ __('Cancel')}}</button>
        <input type="submit" value="{{ __('Invite')}}" id="submit" class="btn  btn-primary">
    </div>
</form>

<script>
    $(function(){
        $("#submit").click(function() {
            var user =  $("#users_list option:selected").length;
            if(user == 0){
            $('#user_validation').removeClass('d-none')
                return false;
            }else{
            $('#user_validation').addClass('d-none')
            }
        });
    });
</script>
