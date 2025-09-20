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

namespace YooKassa\Model\Refund;

use YooKassa\Common\AbstractObject;
use YooKassa\Model\CancellationDetailsInterface;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель RefundCancellationDetails.
 *
 * Комментарий к статусу `canceled`: кто отменил возврат и по какой причине.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $party Инициатор отмены возврата
 * @property string $reason Причина отмены возврата
 */
class RefundCancellationDetails extends AbstractObject implements CancellationDetailsInterface
{
    /**
     * @var string Инициатор отмены возврата
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Choice(callback: [RefundCancellationDetailsPartyCode::class, 'getValidValues'])]
    private string $_party = '';

    /**
     * @var string Причина отмены возврата
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Choice(callback: [RefundCancellationDetailsReasonCode::class, 'getValidValues'])]
    private string $_reason = '';

    /**
     * Возвращает участника процесса возврата, который принял решение об отмене транзакции.
     *
     * @return string Инициатор отмены возврата
     */
    public function getParty(): string
    {
        return $this->_party;
    }

    /**
     * Возвращает причину отмены возврата.
     *
     * @return string Причина отмены возврата
     */
    public function getReason(): string
    {
        return $this->_reason;
    }

    /**
     * Устанавливает участника процесса возврата, который принял решение об отмене транзакции.
     *
     * @param string|null $party
     *
     * @return self
     */
    public function setParty(?string $party = null): self
    {
        $this->_party = $this->validatePropertyValue('_party', $party);
        return $this;
    }

    /**
     * Устанавливает причину отмены возврата.
     *
     * @param string|null $reason
     *
     * @return self
     */
    public function setReason(?string $reason = null): self
    {
        $this->_reason = $this->validatePropertyValue('_reason', $reason);
        return $this;
    }
}
