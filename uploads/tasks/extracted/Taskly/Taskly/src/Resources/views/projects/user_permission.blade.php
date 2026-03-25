<form method="post" action="{{ route('projects.user.permission.store',[$currentWorkspace->slug,$project->id,$user->id]) }}">
    @csrf
    @include('projects.project_permission')
    <div class="modal-footer">
       <button type="button" class="btn  btn-light" data-bs-dismiss="modal">{{ __('Cancel')}}</button>
         <input type="submit" value="{{ __('Save Changes')}}" class="btn  btn-primary">
    </div>
</form>
