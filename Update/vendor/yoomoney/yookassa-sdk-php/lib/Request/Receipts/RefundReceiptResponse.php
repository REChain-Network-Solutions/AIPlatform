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

namespace YooKassa\Request\Receipts;

use YooKassa\Common\Exceptions\InvalidPropertyValueException;
use YooKassa\Common\Exceptions\InvalidPropertyValueTypeException;
use YooKassa\Helpers\TypeCast;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс описывающий чек, привязанный к возврату.
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $refund_id Идентификатор возврата в ЮKassa.
 * @property string $refundId Идентификатор возврата в ЮKassa.
 */
class RefundReceiptResponse extends AbstractReceiptResponse
{
    /** Длина идентификатора возврата */
    public const LENGTH_REFUND_ID = 36;

    /**
     * @var string|null Идентификатор возврата
     */
    #[Assert\Type('string')]
    #[Assert\Length(exactly: self::LENGTH_REFUND_ID)]
    private ?string $_refund_id = null;

    /**
     * Установка свойств, присущих конкретному объекту.
     */
    public function setSpecificProperties(array $receiptData): void
    {
        $this->setRefundId($this->getObjectId());
    }

    /**
     * Возвращает идентификатор возврата.
     *
     * @return string|null Идентификатор возврата
     */
    public function getRefundId(): ?string
    {
        return $this->_refund_id;
    }

    /**
     * Устанавливает идентификатор возврата в ЮKassa.
     *
     * @param string|null $refund_id Идентификатор возврата в ЮKassa
     *
     * @return self
     */
    public function setRefundId(?string $refund_id): self
    {
        $this->_refund_id = $this->validatePropertyValue('_refund_id', $refund_id);
        return $this;
    }
}
