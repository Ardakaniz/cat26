const grid = document.querySelector(".grid-wrapper");
const labels = [
    "soundscapes",
    "La valse du renard et du pic-vert",
    "airblade",
    "Today I saw a snail without a tail",
    "<3",
    "TAKE MY SIX HANDS, WE'RE GOING TO THE SEVEN CLUB",
    "honni soit qui mal y pense",
    "music is healing",
    "スピードバニー",
    "I can’t see without my glasses",
    "Eat my spinal cord",
    "Insgeheim liebe ich jeden",
    "U+262E",
    "scrmbld",
    "just keep going!!",
    "<!DOCTYPE ...>",
    "Retry! Retry! Retry!",
    "Sirinata isulana",
    "digital playground VOL.2",
    "METRO AUDIO STEREO",
    "hermana para siempre",
    "☆🐟",
    "Laserflip",
    "2001",
    "DETERMINATIONMAXXING",
    "cathéter au goudron",
    "PUNJABI PAPAD",
    "amen tek",
    "Up To Something",
    "Down To Nothing",
    "I hope you fall in love with being alive again",
];
const today = 5;

let minHeight = null;
for (let i = 0; i < 31; i++) {
    const hidden = i >= today;

    const item = document.createElement("div");
    item.classList.add("grid-item");

    const label = document.createElement("div");
    label.className = "label";
    label.textContent = labels[i];
    if (hidden) label.style.color = "white";
    item.append(label);

    const imgStatic = document.createElement("img");
    imgStatic.src = hidden ? "assets/todo.png" : `assets/covers/d${i}.png`;
    imgStatic.classList.add("static-img");
    item.append(imgStatic);

    if (!hidden) {
        item.classList.add("activable");

        const imgDyn = document.createElement("img");
        imgDyn.src = `assets/covers/d${i}.gif`;
        imgDyn.classList.add("dyn-img");
        item.append(imgDyn);

        const sound = document.createElement("audio");
        sound.src = `assets/sounds/d${i}.wav`;
        sound.loop = true;
        sound.volume = 0.25;
        item.append(sound);

        item.addEventListener("click", () => {
            item.classList.add("activated");
            sound.play();

            if (!document.fullscreenElement)
                document.getElementsByTagName("main")[0].requestFullscreen();
        });
    }

    grid.append(item);
}
