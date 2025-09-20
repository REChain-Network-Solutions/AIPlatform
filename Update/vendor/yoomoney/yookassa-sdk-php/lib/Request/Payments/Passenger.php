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

namespace YooKassa\Request\Payments;

use Exception;
use YooKassa\Common\AbstractObject;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель PaymentsRequest.
 *
 * Данные пассажира.
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $firstName Имя пассажира
 * @property string $first_name Имя пассажира
 * @property string $lastName Фамилия пассажира
 * @property string $last_name Фамилия пассажира
 */
class Passenger extends AbstractObject implements PassengerInterface
{
    /**
     * Имя пассажира. Необходимо использовать латинские буквы, например SERGEI.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: 64)]
    #[Assert\Length(min: 1)]
    private ?string $_first_name = null;

    /**
     * Фамилия пассажира. Необходимо использовать латинские буквы, например IVANOV.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Length(max: 64)]
    #[Assert\Length(min: 1)]
    private ?string $_last_name = null;


    /**
     * Возвращает first_name.
     *
     * @return string|null
     */
    public function getFirstName(): ?string
    {
        return $this->_first_name;
    }

    /**
     * Устанавливает first_name.
     *
     * @param string|null $value Имя пассажира. Необходимо использовать латинские буквы, например SERGEI.
     *
     * @return self
     */
    public function setFirstName(?string $value = null): self
    {
        $this->_first_name = $this->validatePropertyValue('_first_name', $value);
        return $this;
    }

    /**
     * Возвращает last_name.
     *
     * @return string|null
     */
    public function getLastName(): ?string
    {
        return $this->_last_name;
    }

    /**
     * Устанавливает last_name.
     *
     * @param string|null $value Фамилия пассажира. Необходимо использовать латинские буквы, например IVANOV.
     *
     * @return self
     */
    public function setLastName(?string $value = null): self
    {
        $this->_last_name = $this->validatePropertyValue('_last_name', $value);
        return $this;
    }
}
