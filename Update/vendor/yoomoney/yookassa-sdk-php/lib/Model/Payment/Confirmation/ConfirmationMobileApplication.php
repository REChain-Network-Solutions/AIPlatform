<?php

/*
 * The MIT License
 *
 * Copyright (c) 2025 "YooMoney", NBСO LLC
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

namespace YooKassa\Model\Payment\Confirmation;

use YooKassa\Model\Payment\ConfirmationType;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель ConfirmationMobileApplication.
 *
 * Сценарий, при котором необходимо отправить плательщика на веб-страницу ЮKassa или партнера для
 * подтверждения платежа.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $confirmationUrl URL на который необходимо перенаправить плательщика для подтверждения оплаты
 * @property string $confirmation_url URL на который необходимо перенаправить плательщика для подтверждения оплаты
 */
class ConfirmationMobileApplication extends AbstractConfirmation
{
    /**
     * URL на который необходимо перенаправить плательщика для подтверждения оплаты
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    private ?string $_confirmation_url = null;

    public function __construct($data = [])
    {
        parent::__construct($data);
        $this->setType(ConfirmationType::MOBILE_APPLICATION);
    }

    /**
     * Возвращает URL на который необходимо перенаправить плательщика для подтверждения оплаты
     *
     * @return string|null URL на который необходимо перенаправить плательщика для подтверждения оплаты
     */
    public function getConfirmationUrl(): ?string
    {
        return $this->_confirmation_url;
    }

    /**
     * Устанавливает URL на который необходимо перенаправить плательщика для подтверждения оплаты
     *
     * @param string|null $confirmation_url URL на который необходимо перенаправить плательщика для подтверждения оплаты
     *
     * @return self
     */
    public function setConfirmationUrl(?string $confirmation_url = null): self
    {
        $this->_confirmation_url = $this->validatePropertyValue('_confirmation_url', $confirmation_url);
        return $this;
    }
}
