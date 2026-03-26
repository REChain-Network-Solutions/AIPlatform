@if (in_array(\Modules\Complaint\Entities\Complaint::MODULE_NAME, user_modules()) && in_array('admin', user_roles()))
<x-setting-menu-item :active="$activeMenu" menu="complaint_settings" :href="route('complaint-settings.index')" :text="__('complaint::modules.complaintSettings')" />
@endif
