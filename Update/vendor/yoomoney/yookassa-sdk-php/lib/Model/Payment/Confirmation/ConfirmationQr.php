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
 * Класс, представляющий модель ConfirmationQr.
 *
 * Сценарий при котором для подтверждения платежа пользователю необходимо просканировать QR-код.
 * От вас требуется сгенерировать QR-код, используя любой доступный инструмент, и отобразить его на странице оплаты.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $confirmationData URL для создания QR-кода
 * @property string $confirmation_data URL для создания QR-кода
 */
class ConfirmationQr extends AbstractConfirmation
{
    /**
     * Данные для генерации QR-кода.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    private ?string $_confirmation_data = null;

    public function __construct(?array $data = [])
    {
        parent::__construct($data);

        $this->setType(ConfirmationType::QR);
    }

    /**
     * Возвращает данные для генерации QR-кода.
     *
     * @return string|null
     */
    public function getConfirmationData(): ?string
    {
        return $this->_confirmation_data;
    }

    /**
     * Устанавливает данные для генерации QR-кода.
     *
     * @param string|null $confirmation_data Данные для генерации QR-кода.
     *
     * @return self
     */
    public function setConfirmationData(?string $confirmation_data = null): self
    {
        $this->_confirmation_data = $this->validatePropertyValue('_confirmation_data', $confirmation_data);
        return $this;
    }
}
