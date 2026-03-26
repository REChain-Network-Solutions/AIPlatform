<?php defined('BASEPATH') or exit('No direct script access allowed'); ?>
<?php init_head(); ?>
<div id="wrapper">
	<div class="content">
		<div class="row">
			
			<div class="col-md-12" >
				<div class="panel_s">
					
					<div class="panel-body">
						<div class="row mb-5">

							<div class="col-md-3">
								<h4 class="no-margin"><i class="fa fa-plus" aria-hidden="true"></i>  <?php echo new_html_entity_decode($header); ?> </h4>
							</div>
							<!-- status -->
							<div class="col-md-9">
								<div class="sw-main sw-theme-arrows pull-right">

									<!-- SmartWizard html -->
									<ul class="nav nav-tabs step-anchor">
										<li class="active"><a href="<?php echo admin_url('report_builder/data_source_manage/'.$report_template_id.'?group=relationships'); ?>"><i class="fa fa-dashboard" aria-hidden="true"></i>  <?php echo _l('rb_data_source'); ?></a></li>
										<li class="active"><a href="#"><i class="fa fa-calculator" aria-hidden="true"></i>  <?php echo _l('rb_groups'); ?></a></li>
										<li class="active "><a href="#"><i class="fa fa-plus" aria-hidden="true"></i>  <?php echo _l('rb_subtotals'); ?></a></li>
										<li class=""><a href="<?php echo admin_url('report_builder/column_manage/'.$report_template_id.'?group=column'); ?>"><i class="fa fa-th-large" aria-hidden="true"></i>  <?php echo _l('rb_columns'); ?></a></li>
										<li class=""><a href="<?php echo admin_url('report_builder/cell_formatting/'.$report_template_id); ?>"><i class="fa fa-list-alt" aria-hidden="true"></i>  <?php echo _l('rb_cells'); ?></a></li>
										<li class=""><a href="#"><i class="fa fa-cog menu-icon menu-icon" aria-hidden="true"></i>  <?php echo _l('rb_settings'); ?></a></li>
									</ul>
								</div>

							</div>
							<!-- status -->
							
						</div>

						<!-- start tab -->
						<div class="modal-body">
							<div class="tab-content">
								<div class="row">

									<?php echo form_open(admin_url('report_builder/add_subtotals/'.$report_template_id), array('id' => 'add_subtotals')); ?>
									<div class="row">
										<div class="col-md-12">
											<label for="contracts_view" class="pt-2"></label>
											<div class="form-group">
												<input data-can-view="" type="checkbox" class="capability" id="allow_subtotal" name="allow_subtotal" <?php if($allow_subtotal == 'yes'){ echo "checked";}else{ echo "";} ?>>
												<label for="allow_subtotal" class="pt-2">
													<?php echo _l('rb_allow_subtotals'); ?>               
												</label>
											</div>
										</div>

										<div class="col-md-12">
											<div class="form-group">
												<label for="group_by" class="control-label"><?php echo _l('rb_group_by'); ?></label>
												<select name="group_by[]" class="form-control selectpicker" multiple="true" id="group_by" data-actions-box="true" data-width="100%" data-none-selected-text="<?php echo _l('dropdown_non_selected_tex'); ?>" data-live-search="true" disabled> 
													<?php foreach ($templates_selected_column as $key => $selected_column) { ?>

														<?php 
															$selected = '';
															if($selected_column['group_by'] == 'yes'){
																$selected .= ' selected';
															}
														 ?>

														<option value="<?php echo new_html_entity_decode($selected_column['id']); ?>" <?php echo new_html_entity_decode($selected); ?> ><?php  echo new_html_entity_decode(_l('tbl'.$selected_column['table_name'].'_'.$selected_column['field_name'])); ?><small> (<?php  echo new_html_entity_decode(_l('tbl'.$selected_column['table_name'])); ?>)</small></option>
													<?php } ?>
												</select>
											</div>
										</div>

										<div class="col-md-12">
											<?php echo render_select('function_name',rb_subtotals_data(),array('name','label'),'rb_function_name', $function_name, [], [], '', '', false); ?>
											
										</div>

										<div class="col-md-5">
											<label><?php echo _l('rb_columns'); ?></label>
											<select name="from[]" id="lstview" class="form-control" size="15" multiple="multiple">

												<?php  foreach ($columns as $column) { ?>
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
											<label><?php echo _l('rb_affected_columns'); ?></label>
											<select name="to[]" id="lstview_to" class="form-control" size="15" multiple="multiple">
												<?php  foreach ($affected_columns as $selected_column) { ?>
													<option value="<?php echo new_html_entity_decode($selected_column['name']); ?>"><?php echo new_html_entity_decode($selected_column['label']); ?></option>
												<?php } ?>
											</select>
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


								</div>
							</div>
						</div>

							<div class="modal-footer">
								<a href="<?php echo admin_url('report_builder/report_manage'); ?>"  class="btn btn-default mr-2 "><?php echo _l('close'); ?></a>
								<a href="<?php echo admin_url('report_builder/group_manage/'.$report_template_id.'?group=group_by'); ?>"  class="btn btn-default mr-2 "><?php echo _l('rb_back'); ?></a>
								<button class="btn btn-info mr-2 data_source_next"><?php echo _l('rb_next'); ?></button>
							</div>

						</div>
					</div>
				</div>

			</div>
		</div>
		<?php echo form_hidden('report_template_id',$report_template_id); ?>


		<?php init_tail(); ?>
		<?php 

		require('modules/report_builder/assets/js/templates/subtotals/subtotal_manage_js.php');
	

		?>
	</body>

	</html>
