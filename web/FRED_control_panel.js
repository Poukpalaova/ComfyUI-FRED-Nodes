app.registerExtension({
    name: "comfy.advanced_denoise_node",
    init() {
        LiteGraph.registerNodeType("Custom Nodes/Image Processing/AdvancedDenoise", {
            title: "Advanced Denoise",
            onNodeCreated(node) {
                const container = document.createElement("div");
                
                // Mode selector buttons
                const modes = ["Face", "Upper Body", "Full Body", "No Human"];
                modes.forEach(mode => {
                    const btn = document.createElement("button");
                    btn.textContent = mode;
                    btn.style.margin = "2px";
                    btn.onclick = () => node.setProperty("mode", mode);
                    container.appendChild(btn);
                });
                
                // Parameter groups
                const createGroup = (title, params) => {
                    const group = document.createElement("div");
                    group.style.margin = "10px 0";
                    group.innerHTML = `<strong>${title}</strong>`;
                    params.forEach(param => {
                        const input = document.createElement("input");
                        input.type = "number";
                        input.value = node.properties[param];
                        input.onchange = (e) => node.setProperty(param, parseFloat(e.target.value));
                        group.appendChild(input);
                    });
                    return group;
                };

                container.appendChild(createGroup("Zone 1", ["denoise_zone_1", "guidance_zone_1"]));
                // Add other groups similarly
                
                node.addDOMWidget("controls", container);
            }
        });
    }
});