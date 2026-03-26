@if (in_array(\Modules\Feedback\Entities\Feedback::MODULE_NAME, user_modules()) && in_array('admin', user_roles()))
<x-setting-menu-item :active="$activeMenu" menu="feedback_settings" :href="route('feedback-settings.index')" :text="__('feedback::modules.feedbackSettings')" />
@endif
