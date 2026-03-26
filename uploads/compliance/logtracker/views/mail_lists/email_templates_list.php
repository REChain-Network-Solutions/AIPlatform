<div class="col-md-12">
	<h4 class="bold well email-template-heading">
		<?php echo _l('logtracker_email_templates'); ?>
		<?php if ($hasPermissionEdit) { ?>
			<a href="<?php echo admin_url('emails/disable_by_type/logtracker'); ?>"
				class="pull-right mleft5 mright25"><small>
					<?php echo _l('disable_all'); ?>
				</small></a>
			<a href="<?php echo admin_url('emails/enable_by_type/logtracker'); ?>" class="pull-right"><small>
					<?php echo _l('enable_all'); ?>
				</small></a>
		<?php } ?>

	</h4>
	<div class="table-responsive">
		<table class="table table-bordered">
			<thead>
				<tr>
					<th>
						<?php echo _l('email_templates_table_heading_name'); ?>
					</th>
				</tr>
			</thead>
			<tbody>
				<?php foreach ($logtracker as $logtracker) { ?>
					<tr>
						<td class="<?php if (0 == $logtracker['active']) {
							echo 'text-throught';
						} ?>">
							<a href="<?php echo admin_url('emails/email_template/' . $logtracker['emailtemplateid']); ?>">
								<?php echo $logtracker['name']; ?>
							</a>
							<?php if (ENVIRONMENT !== 'production') { ?>
								<br /><small>
									<?php echo $logtracker['slug']; ?>
								</small>
							<?php } ?>
							<?php if ($hasPermissionEdit) { ?>
								<a href="<?php echo admin_url('emails/' . ('1' == $logtracker['active'] ? 'disable/' : 'enable/') . $logtracker['emailtemplateid']); ?>"
									class="pull-right"><small>
										<?php echo _l(1 == $logtracker['active'] ? 'disable' : 'enable'); ?>
									</small></a>
							<?php } ?>
						</td>
					</tr>
				<?php } ?>
			</tbody>
		</table>
	</div>
</div>