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

namespace YooKassa\Model\Payment;

use YooKassa\Common\AbstractObject;
use YooKassa\Model\Invoice\InvoiceInterface;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель PaymentInvoiceDetails.
 *
 * Данные о выставленном счете, в рамках которого проведен платеж.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $id Идентификатор счета в ЮKassa.
 */
class PaymentInvoiceDetails extends AbstractObject
{
    /**
     * Идентификатор счета в ЮKassa.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    #[Assert\Length(max: InvoiceInterface::MAX_LENGTH_ID)]
    #[Assert\Length(min: InvoiceInterface::MIN_LENGTH_ID)]
    private ?string $_id = null;

    /**
     * Возвращает id.
     *
     * @return string|null Идентификатор счета в ЮKassa.
     */
    public function getId(): ?string
    {
        return $this->_id;
    }

    /**
     * Устанавливает id.
     *
     * @param string|null $id Идентификатор счета в ЮKassa.
     *
     * @return self
     */
    public function setId(?string $id = null): self
    {
        $this->_id = $this->validatePropertyValue('_id', $id);
        return $this;
    }

}

