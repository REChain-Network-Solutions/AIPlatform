<?php defined('BASEPATH') or exit('No direct script access allowed'); ?>

<?php echo form_open_multipart(admin_url('report_builder/add_label_cell_type/'.$report_template_id), array('id'=>'add_label_cell_type')); ?>

<div class="row">
	<div class="col-md-12">
		<h5 class="font-style-italic font-color-red"><?php echo _l('rb_label_cell_type_note'); ?></h5>
	</div>
</div>
<div class="form"> 
	<div id="label_cell_type_hs" class="hot handsontable htColumnHeaders">
	</div>
	<?php echo form_hidden('label_cell_type_hs'); ?>
</div>
<div class="row">
	<div class="col-md-12">
		<div class="modal-footer">
			<?php if(has_permission('report_builder', '', 'create') || has_permission('report_builder', '', 'edit')){ ?>
				<button class="btn btn-info pull-right add_label_cell_type"><?php echo _l('submit'); ?></button>
			<?php } ?>
		</div>
	</div>
</div>
<?php echo form_close(); ?>

</body>
</html>
