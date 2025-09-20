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

namespace YooKassa\Model\Refund\RefundMethod;

use YooKassa\Model\Refund\RefundMethodType;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель RefundMethodSbp.
 *
 * Возврат через СБП (Система быстрых платежей ЦБ РФ).
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string|null $sbp_operation_id Идентификатор операции в СБП (НСПК).
 * @property string|null $sbpOperationId Идентификатор операции в СБП (НСПК).
 */
class RefundMethodSbp extends AbstractRefundMethod
{
    /**
     * Идентификатор операции в СБП (НСПК). Пример: `1027088AE4CB48CB81287833347A8777` Обязательный параметр для возвратов в статусе ~`succeeded`. В остальных случаях может отсутствовать.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    private ?string $_sbp_operation_id = null;

    public function __construct(?array $data = [])
    {
        parent::__construct($data);
        $this->setType(RefundMethodType::SBP);
    }

    /**
     * Возвращает sbp_operation_id.
     *
     * @return string|null
     */
    public function getSbpOperationId(): ?string
    {
        return $this->_sbp_operation_id;
    }

    /**
     * Устанавливает sbp_operation_id.
     *
     * @param string|null $sbp_operation_id Идентификатор операции в СБП (НСПК).
     *
     * @return self
     */
    public function setSbpOperationId(?string $sbp_operation_id = null): self
    {
        $this->_sbp_operation_id = $this->validatePropertyValue('_sbp_operation_id', $sbp_operation_id);
        return $this;
    }

}
