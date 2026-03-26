<?php defined('BASEPATH') or exit('No direct script access allowed'); ?>

<div>
	
	<div class="clearfix"></div>
	<br>
	<div>
		<?php echo form_open(admin_url('report_builder/add_relationship'), array('id' => 'add_relationship')); ?>
		<?php 
		echo form_hidden('isedit');
		 ?>
			<div class="modal-body">
				<div class="tab-content">
					<div class="row">
						<div class="col-md-12 relationship_data">
							<?php echo form_hidden('templates_id',$report_template_id); ?>

							<?php echo new_html_entity_decode($relationship_row_template); ?>
							

							<div class="row hide">
								
								<div class="col-md-12"> 
									<?php echo render_textarea('query_string','query_string'); ?>
								</div>
							</div>

						</div>
						<div id="removed-items"></div>

					</div>
				</div>
			</div>
			<div class="modal-footer">
		<?php  if(is_admin() || has_permission('report_builder','','create')) { ?>

				<button type="submit" class="btn btn-info"><?php echo _l('submit'); ?></button>
		<?php } ?>

			</div>
		<?php echo form_close(); ?>

	</div>
	<div id="modal_wrapper"></div>
</div>
