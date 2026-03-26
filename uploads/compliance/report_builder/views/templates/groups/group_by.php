<?php defined('BASEPATH') or exit('No direct script access allowed'); ?>

<?php echo form_open(admin_url('report_builder/add_group_by_columns/'.$report_template_id), array('id' => 'add_group_by_columns')); ?>
<div class="row">
	<div class="col-md-12">
		<div class="col-md-5">
			<label><?php echo _l('rb_available_fields'); ?></label>
			<select name="from[]" id="lstview" class="form-control" size="15" multiple="multiple">

				<?php  foreach ($column_of_table as $column) { ?>
					<option value="<?php echo new_html_entity_decode($column['name']); ?>"><?php echo new_html_entity_decode($column['label']); ?></option>
				<?php } ?>
			</select>
		</div>

		<div class="col-md-2">
			<label></label>
			<br>
			<button type="button" id="lstview_undo" class="btn btn-danger btn-block">undo</button>
			<button type="button" id="lstview_rightAll" class="btn btn-default btn-block"><i class="glyphicon glyphicon-forward"></i></button>
			<button type="button" id="lstview_rightSelected" class="btn btn-default btn-block"><i class="glyphicon glyphicon-chevron-right"></i></button>
			<button type="button" id="lstview_leftSelected" class="btn btn-default btn-block"><i class="glyphicon glyphicon-chevron-left"></i></button>
			<button type="button" id="lstview_leftAll" class="btn btn-default btn-block"><i class="glyphicon glyphicon-backward"></i></button>
			<button type="button" id="lstview_redo" class="btn btn-warning btn-block">redo</button>
		</div>

		<div class="col-md-5">
			<label><?php echo _l('rb_group_by'); ?></label>
			<select name="to[]" id="lstview_to" class="form-control" size="15" multiple="multiple">
				<?php  foreach ($selected_columns as $selected_column) { ?>
					<option value="<?php echo new_html_entity_decode($selected_column['name']); ?>"><?php echo new_html_entity_decode($selected_column['label']); ?></option>
				<?php } ?>
			</select>
		</div>
	</div>
</div>
<div class="row">
	<div class="col-md-12">
		<div class="modal-footer">
			<button type="submit" class="btn btn-info"><?php echo _l('submit'); ?></button>
		</div>
	</div>
</div>
<?php echo form_close(); ?>

</body>


</html>
