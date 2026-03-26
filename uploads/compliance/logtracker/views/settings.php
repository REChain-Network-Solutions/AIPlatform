<p class="tw-font-semibold tw-text-lg">
	<?php echo _l('log_level_color_options') ?>
</p>
<div class="row">
	<div class="col-md-4">
		<?php echo render_color_picker('settings[date_color]', _l('date'), get_option('date_color')) ?>
	</div>
	<div class="col-md-4">
		<?php echo render_color_picker('settings[all_color]', _l('all'), get_option('all_color')) ?>
	</div>
	<div class="col-md-4">
		<?php echo render_color_picker('settings[critical_color]', _l('critical'), get_option('critical_color')) ?>
	</div>
	<div class="col-md-4">
		<?php echo render_color_picker('settings[error_color]', _l('error'), get_option('error_color')) ?>
	</div>
	<div class="col-md-4">
		<?php echo render_color_picker('settings[debug_color]', _l('debug'), get_option('debug_color')) ?>
	</div>
	<div class="col-md-4">
		<?php echo render_color_picker('settings[info_color]', _l('info'), get_option('info_color')) ?>
	</div>
</div>
<?php if (is_admin()) { ?>
	<hr class="">
	<p class="tw-font-semibold tw-text-lg">
		<?php echo _l('enviornment_mode') ?>
	</p>
	<div class="row">
		<div class="col-md-12">
			<?php render_yes_no_option('enviornment_mode', '', '', _l('development'), _l('production'), 'development', 'production'); ?>
		</div>
	</div>
	<?php echo displayEnvironmentMessage(); ?>
<?php } ?>