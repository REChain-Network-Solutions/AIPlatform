@extends('ai-social-media::automation-steps.layout')

@section('yield_content')
	<div class="mb-8 border-b pb-6">
		<h3 class="mb-5 flex flex-wrap items-center justify-between gap-3">
			@lang('Review Your Content')
		</h3>
		<p>
			@lang('Start by selecting a company and product or create new ones at BrandCenter in a few clicks.')
		</p>
	</div>

	<form
		class="flex flex-col gap-6"
		id="stepsForm"
		action="{{ route('dashboard.user.automation.step.last') }}"
		method="POST"
	>
		@csrf
		<input
			type="hidden"
			name="platform_id"
			value="{{ $platform_id }}"
		/>
		<input
			type="hidden"
			name="company_id"
			value="{{ $company_id }}"
		/>
		@foreach ($product_id as $pid)
			<input
				type="hidden"
				name="product_id[]"
				value="{{ $pid }}"
			>
		@endforeach
		<input
			type="hidden"
			name="camp_id"
			value="{{ $camp_id }}"
		/>
		<input
			type="hidden"
			name="camp_target"
			value="{{ $camp_target }}"
		/>
		@foreach ($topics as $topic)
			<input
				type="hidden"
				name="topics[]"
				value="{{ $topic }}"
			>
		@endforeach
		<input
			type="hidden"
			name="seo"
			value="{{ $seo ?? false }}"
		/>
		<input
			type="hidden"
			name="is_img"
			value="{{ $is_img ?? false }}"
		/>
		<input
			type="hidden"
			name="tone"
			value="{{ $tone ?? false }}"
		/>
		<input
			type="hidden"
			name="num_res"
			value="{{ $num_res ?? false }}"
		/>
		<input
			type="hidden"
			name="vis_format"
			value="{{ $vis_format ?? false }}"
		/>
		<input
			type="hidden"
			name="vis_ratio"
			value="{{ $vis_ratio ?? false }}"
		/>
		<input
			id="cam_injected_name"
			type="hidden"
			name="cam_injected_name"
			value="{{ $cam_injected_name }}"
		/>

		<input
			type="hidden"
			name="step"
			value="6"
		/>

		@if($platform['key'] === \App\Extensions\AISocialMedia\System\Enums\Platform::instagram->value)

			<div>

				<div class="mb-2">
					<x-forms.input
						id="generate_image_prompt"
						tooltip="{{ __('Images are generated using the DALL-E model. Please check if you have any credits.') }}"
						label="{{ __('Can you generate images with artificial intelligence?') }}"
						size="lg"
						type="text"
						name="generate_image_prompt"
						placeholder="{{ __('Enter image prompt') }}"
						>
						<x-slot:label-extra>
							<div
									class="font-['Golos Text'] text-xs font-semibold leading-relaxed text-purple-950"
									style="cursor: pointer;"
							>
								<div
										class="lds-dual-ring hidden"
										id="lds-dual-ring1"
								></div>
								<svg id="generate_image_btn" class="generate" width="35" height="35" viewBox="0 0 35 35" fill="none" xmlns="http://www.w3.org/2000/svg" > <path fill-rule="evenodd" clip-rule="evenodd" d="M25.3483 15.0849L23.9529 15.3678C23.2806 15.5042 22.6635 15.8356 22.1785 16.3207C21.6934 16.8057 21.362 17.4229 21.2256 18.0951L20.9426 19.4905C20.9147 19.6301 20.8392 19.7556 20.7291 19.8458C20.619 19.936 20.4811 19.9853 20.3388 19.9853C20.1964 19.9853 20.0585 19.936 19.9484 19.8458C19.8383 19.7556 19.7629 19.6301 19.7349 19.4905L19.4519 18.0951C19.3156 17.4228 18.9843 16.8056 18.4992 16.3205C18.0142 15.8355 17.3969 15.5041 16.7246 15.3678L15.3292 15.0849C15.19 15.0564 15.0648 14.9807 14.9749 14.8706C14.885 14.7604 14.8359 14.6226 14.8359 14.4805C14.8359 14.3384 14.885 14.2006 14.9749 14.0905C15.0648 13.9803 15.19 13.9046 15.3292 13.8762L16.7246 13.5932C17.3969 13.4569 18.0142 13.1255 18.4992 12.6405C18.9843 12.1554 19.3156 11.5382 19.4519 10.8659L19.7349 9.47048C19.7629 9.33094 19.8383 9.20539 19.9484 9.11519C20.0585 9.025 20.1964 8.97571 20.3388 8.97571C20.4811 8.97571 20.619 9.025 20.7291 9.11519C20.8392 9.20539 20.9147 9.33094 20.9426 9.47048L21.2256 10.8659C21.362 11.5381 21.6934 12.1553 22.1785 12.6403C22.6635 13.1254 23.2806 13.4568 23.9529 13.5932L25.3483 13.8762C25.4876 13.9046 25.6127 13.9803 25.7026 14.0905C25.7925 14.2006 25.8416 14.3384 25.8416 14.4805C25.8416 14.6226 25.7925 14.7604 25.7026 14.8706C25.6127 14.9807 25.4876 15.0564 25.3483 15.0849ZM15.1421 22.1572L14.763 22.2342C14.2954 22.3291 13.8662 22.5595 13.5288 22.8969C13.1915 23.2342 12.961 23.6634 12.8662 24.131L12.7892 24.5102C12.7679 24.6164 12.7105 24.7119 12.6267 24.7806C12.5429 24.8492 12.438 24.8867 12.3297 24.8867C12.2214 24.8867 12.1164 24.8492 12.0326 24.7806C11.9488 24.7119 11.8914 24.6164 11.8701 24.5102L11.7931 24.131C11.6983 23.6634 11.4678 23.2342 11.1305 22.8969C10.7931 22.5595 10.3639 22.3291 9.89634 22.2342L9.51718 22.1572C9.41098 22.1359 9.31543 22.0785 9.24678 21.9947C9.17813 21.911 9.14062 21.806 9.14062 21.6977C9.14062 21.5894 9.17813 21.4844 9.24678 21.4006C9.31543 21.3169 9.41098 21.2594 9.51718 21.2382L9.89634 21.1612C10.3639 21.0663 10.7931 20.8358 11.1305 20.4985C11.4678 20.1611 11.6983 19.7319 11.7931 19.2644L11.8701 18.8852C11.8914 18.779 11.9488 18.6834 12.0326 18.6148C12.1164 18.5461 12.2214 18.5087 12.3297 18.5087C12.438 18.5087 12.5429 18.5461 12.6267 18.6148C12.7105 18.6834 12.7679 18.779 12.7892 18.8852L12.8662 19.2644C12.961 19.7319 13.1915 20.1611 13.5288 20.4985C13.8662 20.8358 14.2954 21.0663 14.763 21.1612L15.1421 21.2382C15.2483 21.2594 15.3439 21.3169 15.4125 21.4006C15.4812 21.4844 15.5187 21.5894 15.5187 21.6977C15.5187 21.806 15.4812 21.911 15.4125 21.9947C15.3439 22.0785 15.2483 22.1359 15.1421 22.1572Z" fill="url(#paint0_linear_2401_1456)" /> <defs> <linearGradient id="paint0_linear_2401_1456" x1="25.8416" y1="16.9312" x2="8.97735" y2="14.9877" gradientUnits="userSpaceOnUse" > <stop stop-color="#8D65E9" /> <stop offset="0.483" stop-color="#5391E4" /> <stop offset="1" stop-color="#6BCD94" /> </linearGradient> </defs> </svg>
							</div>
						</x-slot:label-extra>
					</x-forms.input>
				</div>

				<div class="max-w-md mx-auto">
					<div
						id="dropzone"
						class="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center cursor-pointer hover:border-blue-400"
						onclick="fileInput.click()"
					>
						<div id="placeholder" class="flex flex-col items-center space-y-2 {{ old('image') ? 'hidden': '' }}">
							<svg
								class="w-12 h-12 text-gray-400"
								xmlns="http://www.w3.org/2000/svg"
								fill="none"
								viewBox="0 0 24 24"
								stroke="currentColor"
							>
								<x-tabler-upload class="w-12 h-12 text-gray-400" />
							</svg>
							<p class="text-gray-600 {{ old('image') ? 'hidden': '' }}">Fotoğraf yüklemek için tıklayın</p>
						</div>
						<img id="preview" src="{{ old('image') ? '/uploads/'.old('image') : '' }}" class="{{ old('image') ? '' : 'hidden' }} w-full h-64 object-cover rounded-lg" />
					</div>

					<!-- Gizli Dosya Seçici -->
					<input
						id="fileInput"
						type="file"
						class="hidden"
						accept="image/*"
						onchange="handleFileChange(event)"
					/>
					<input type="hidden" name="image" id="image" value="{{ old('image') }}">

					<!-- Yükleme Durumu -->
					<p id="uploadStatus" class="text-center text-gray-500 mt-4 hidden">Yükleniyor...</p>
				</div>

				<div class="flex">


					<div
						class="grow md:flex-1 mt-1"
						id="thePost"
					>
						<div id="loader" class="mt-4 flex items-center justify-center bg-white bg-opacity-75 z-50">
							<div class="text-center">
								<!-- Dönen Yükleme Simgesi -->
								<svg
									class="animate-spin h-10 w-10 text-blue-500 mx-auto mb-4"
									xmlns="http://www.w3.org/2000/svg"
									fill="none"
									viewBox="0 0 24 24"
								>
									<circle
										class="opacity-25"
										cx="12"
										cy="12"
										r="10"
										stroke="currentColor"
										stroke-width="4"
									></circle>
									<path
										class="opacity-75"
										fill="currentColor"
										d="M4 12a8 8 0 018-8v8z"
									></path>
								</svg>
								<!-- Yükleme Mesajı -->
								<p class="text-lg font-medium text-gray-700">{{ __('Content is being generated, please wait...') }}</p>
							</div>
						</div>
					</div>
				</div>

			</div>

		@else
			<div class="flex">


				<div
					class="grow md:flex-1 mt-1"
					id="thePost"
				>
					<div id="loader" class="mt-4 flex items-center justify-center bg-white bg-opacity-75 z-50">
						<div class="text-center">
							<!-- Dönen Yükleme Simgesi -->
							<svg
								class="animate-spin h-10 w-10 text-blue-500 mx-auto mb-4"
								xmlns="http://www.w3.org/2000/svg"
								fill="none"
								viewBox="0 0 24 24"
							>
								<circle
									class="opacity-25"
									cx="12"
									cy="12"
									r="10"
									stroke="currentColor"
									stroke-width="4"
								></circle>
								<path
									class="opacity-75"
									fill="currentColor"
									d="M4 12a8 8 0 018-8v8z"
								></path>
							</svg>
							<!-- Yükleme Mesajı -->
							<p class="text-lg font-medium text-gray-700">{{ __('Content is being generated, please wait...') }}</p>
						</div>
					</div>
				</div>
			</div>
		@endif

		<input type="hidden" name="content" id="content" value="{{ old('content') }}">



		<x-forms.input
			type="checkbox"
			label="{{ __('Send a copy to my email address') }}"
			name="sendMail"
			type="checkbox"
			switcher
		/>


		<x-forms.input
			class="mt-2"
			type="checkbox"
			label="{{ __('Auto generate') }}"
			name="auto_generate"
			type="checkbox"
			switcher
		/>
		<x-alert type="info mt-2">
			{{ __('If you want the content to be generated automatically, you can check this box. The content will be regenerated for each post.') }}
		</x-alert>
		<x-button
			id="reviweNextBtn"
			variant="secondary"
			type="submit"
		>
			{{ __('Next') }}
			<span class="size-7 inline-grid place-items-center rounded-full bg-background text-foreground">
            <x-tabler-chevron-right class="size-4" />
        </span>
		</x-button>
	</form>
@endsection
@push('script')
	<script>


		$('#generate_image_btn').on('click', function () {

			const fileInput = document.getElementById("fileInput");
			const preview = document.getElementById("preview");
			const placeholder = document.getElementById("placeholder");
			const uploadStatus = document.getElementById("uploadStatus");
			const imageInput = document.getElementById("image");
			const generate_image_btn = document.getElementById("generate_image_btn");

			let prompt = $('#generate_image_prompt').val();

			if(prompt === '') {
				toastr.error('Please enter a prompt');
				return;
			}

			$('#lds-dual-ring1').toggleClass('hidden');
			$('.generate').toggleClass('hidden');

			$.ajax({
				url: "/dashboard/user/openai/generate",
				type: 'POST',
				data: {
					"post_type": "ai_image_generator",
					"openai_id": "36",
					"custom_template": "0",
					"image_generator": "openai",
					"image_style": "",
					"image_lighting": "",
					"image_mood": "",
					"image_number_of_images": "1",
					"size": "1024x1024",
					"quality": "standard",
					"description": prompt
				},
				beforeSend: function () {
				},
				success: function (response) {
					if(response.status === 'success') {
						let images = response.images;

						if(images[0]) {
							let image = images[0];

							let output = image.output;

							preview.src = output;

							preview.classList.remove("hidden");

							placeholder.classList.add("hidden");

							imageInput.value = response.nameOfImage;

							$('#lds-dual-ring1').toggleClass('hidden');
							$('.generate').toggleClass('hidden');

						}
					}else {
						toastr.error(response.message);
						$('#lds-dual-ring1').toggleClass('hidden');
						$('.generate').toggleClass('hidden');
					}
				},
				error: function (error) {
					// console.log(error);
					toastr.error(error.responseJSON.message);
					$('#lds-dual-ring1').toggleClass('hidden');
					$('.generate').toggleClass('hidden');
				}
			});
		});
	</script>

	@if($platform['key'] === \App\Extensions\AISocialMedia\System\Enums\Platform::instagram->value)
		<script>
			const fileInput = document.getElementById("fileInput");
			const preview = document.getElementById("preview");
			const placeholder = document.getElementById("placeholder");
			const uploadStatus = document.getElementById("uploadStatus");
			const image = document.getElementById("image");

			async function handleFileChange(event) {
				const file = event.target.files[0];
				if (file && file.type.startsWith("image/")) {
					uploadStatus.classList.remove("hidden");
					placeholder.classList.add("hidden");

					try {
						// FormData oluştur
						const formData = new FormData();
						formData.append("image", file);
						formData.append("_token", "{{ csrf_token() }}");

						// Resmi POST et
						const response = await fetch("{{ route('dashboard.user.automation.upload') }}", {
							method: "POST",
							body: formData,
						});

						if (!response.ok) {
							toast.error("Yükleme sırasında bir hata oluştu.");
						}

						// Sunucudan gelen URL'yi al
						const data = await response.json();
						const imageUrl = data.url; // Sunucunun döndüğü resim URL'si

						image.value = data.path;
						// Önizleme alanını güncelle
						preview.src = imageUrl;
						preview.classList.remove("hidden");
					} catch (error) {
						alert(error.message);
						placeholder.classList.remove("hidden");
					} finally {
						uploadStatus.classList.add("hidden");
					}
				}
			}

		</script>
	@endif
@endpush
