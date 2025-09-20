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

namespace YooKassa\Model\PersonalData;

use YooKassa\Common\AbstractObject;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель PersonalDataCancellationDetails.
 *
 * Комментарий к статусу ~`canceled`: кто и по какой причине аннулировал хранение данных.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 * @property string $party Участник процесса, который принял решение о прекращении хранения персональных данных
 * @property string $reason Причина прекращения хранения персональных данных
 */
class PersonalDataCancellationDetails extends AbstractObject
{
    /**
     * Участник процесса, который принял решение о прекращении хранения персональных данных.
     * Возможное значение:
     * * `yoo_money` — ЮKassa.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Choice(callback: [PersonalDataCancellationDetailsPartyCode::class, 'getValidValues'])]
    #[Assert\Type('string')]
    private ?string $_party = null;

    /**
     * Причина прекращения хранения персональных данных.
     * Возможное значение:
     * * `expired_by_timeout` — истек срок хранения или использования персональных данных.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Choice(callback: [PersonalDataCancellationDetailsReasonCode::class, 'getValidValues'])]
    #[Assert\Type('string')]
    private ?string $_reason = null;

    /**
     * Возвращает участника процесса выплаты, который принял решение о прекращении хранения персональных данных.
     *
     * @return string|null Инициатор отмены о хранения персональных данных.
     */
    public function getParty(): ?string
    {
        return $this->_party;
    }

    /**
     * Устанавливает участника процесса выплаты, который принял решение о прекращении хранения персональных данных.
     *
     * @param string|null $party Участник процесса выплаты, который принял решение о прекращении хранения персональных данных.
     *
     * @return self
     */
    public function setParty(?string $party = null): self
    {
        $this->_party = $this->validatePropertyValue('_party', $party);
        return $this;
    }

    /**
     * Возвращает причину прекращения хранения персональных данных.
     *
     * @return string|null Причина прекращения хранения персональных данных.
     */
    public function getReason(): ?string
    {
        return $this->_reason;
    }

    /**
     * Устанавливает прекращения хранения персональных данных.
     *
     * @param string|null $reason Причина прекращения хранения персональных данных
     *
     * @return self
     */
    public function setReason(?string $reason = null): self
    {
        $this->_reason = $this->validatePropertyValue('_reason', $reason);
        return $this;
    }
}
