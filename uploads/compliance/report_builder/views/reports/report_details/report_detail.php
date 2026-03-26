<?php defined('BASEPATH') or exit('No direct script access allowed'); ?>

<?php init_head(); ?>
<div id="wrapper">
	<div class="content">
		<div class="row">
			<div class="col-md-12">
				<div class="panel_s"> 
					<div class="panel-body">

						<div class="row">
							<div class="col-md-12">
								<h4 class="h4-color no-margin"><i class="fa fa-list-alt menu-icon menu-icon" aria-hidden="true"></i> <?php echo _l('rb_report_detail'); ?></h4>
							</div>
						</div>
						<hr class="hr-color">

						<?php if(has_permission('report_builder', '', 'create')){ ?>
							<div class="_buttons hide">

								<div class="btn-group mleft5">
									<a href="#" class="btn btn-info dropdown-toggle" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false"><?php echo _l('rb_export').' '; ?><span class="caret"></span></a>
									<ul class="dropdown-menu dropdown-menu-right">
										<li class="hidden-xs"><a href="<?php echo admin_url('report_builder/add_data_source'); ?>"  >
											<?php echo _l('rb_export_pdf'); ?>
										</a>
									</li>

									<li class="hidden-xs"><a href="#" onclick="new_job_p(); return false;">
										<?php echo _l('rb_export_xlsx'); ?></a>
									</li>
								</ul>
							</div>

						</div>
						<br>
					<?php } ?>

						<div class="row">
							<div class="col-md-12 text-center">
								<h4><b><?php echo new_html_entity_decode(isset($report_template) ? $report_template->report_name : '') ?></b></h4>
							</div>
						</div>

						<div class="row">
							<div class="col-md-12 justify-content-start">
								<p><?php echo new_html_entity_decode(isset($report_template) ? $report_template->report_header : '') ?></p>
							</div>
						</div>

						<div class="table-responsive">

					<?php if(count($report_result) > 0 ){ ?>
							<?php if($report_group_by){ 

								$this->load->view('report_builder/reports/report_details/report_detail_group_by', $report_result);
							}else{ 

								$this->load->view('report_builder/reports/report_details/report_detail_non_group_by', $report_result);
							 } ?>

						<?php }else{ ?>
							<h4><?php echo _l('rb_no_corresponding_data'); ?></h4>
						<?php } ?>
					</div>

						<div class="row">
							<div class="col-md-12 justify-content-start">
								<p><?php echo new_html_entity_decode(isset($report_template) ? $report_template->report_footer : '') ?></p>
							</div>
						</div>

					</div>
				</div>
			</div>

		</div>
	</div>
</div>
<?php init_tail(); ?>

</body>
</html>
