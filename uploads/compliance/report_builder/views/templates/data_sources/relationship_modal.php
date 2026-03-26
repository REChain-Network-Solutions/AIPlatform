<div class="modal fade" id="relationshipModal">
	<div class="modal-dialog modal-lg">
		<div class="modal-content">
			<div class="modal-header">
				<button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">&times;</span></button>

				<h4 class="modal-title"><?php echo new_html_entity_decode(_l('add_relationship')); ?></h4>
			</div>
			<?php echo form_open(admin_url('report_builder/add_relationship'), array('id' => 'add_relationship')); ?>
			<div class="modal-body">
				<div class="tab-content">
					<div class="row">
						<div class="col-md-12">
							<?php echo form_hidden('templates_id',$template_id); ?>
							
							<div class="row">
								<div class="col-md-4"> 
									<?php echo render_select('left_table',$tables,array('name','name'),'left_table', ''); ?>
								</div>
								<div class="col-md-4"> 
									<?php echo render_select('left_field_1',[],array('id','description'),'left_field_1', ''); ?>
								</div>
								<div class="col-md-4"> 
									<?php echo render_select('left_field_2',[],array('id','description'),'left_field_2', ''); ?>
								</div>
							</div>
							<div class="row">

								<div class="col-md-4"> 
									<?php echo render_select('right_table',$tables,array('name','name'),'right_table', ''); ?>
								</div>
								<div class="col-md-4"> 
									<?php echo render_select('right_field_1',[],array('id','description'),'right_field_1', ''); ?>
								</div>
								<div class="col-md-4"> 
									<?php echo render_select('right_field_2',[],array('id','description'),'right_field_2', ''); ?>
								</div>
							</div>
							<div class="row">
								
								<div class="col-md-12"> 
									<?php echo render_textarea('query_string','query_string'); ?>
								</div>
							</div>

						</div>
					</div>
				</div>
			</div>
			<div class="modal-footer">
				<button type="button" class="btn btn-default close_btn" data-dismiss="modal"><?php echo _l('close'); ?></button>
				<button type="submit" class="btn btn-info"><?php echo _l('submit'); ?></button>
			</div>

		</div>

		<?php echo form_close(); ?>
	</div>
</div>
</div>
<?php require('modules/report_builder/assets/js/templates/data_source/relationships_modal_js.php'); ?>